from flask import Flask, request, redirect, url_for, render_template, jsonify
import os
import datetime
import subprocess
import boto3
from botocore.exceptions import NoCredentialsError
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import json
import requests
import tempfile
import io
from dotenv import load_dotenv


app = Flask(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.efficientnet_b4(pretrained=False)
num_classes = 5
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(model.classifier[1].in_features, num_classes)
)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()
model.to(device)

class_labels = ['FACESHAPE_HEART', 'FACESHAPE_OBLONG', 'FACESHAPE_OVAL', 'FACESHAPE_ROUND', 'FACESHAPE_SQUARE']


# 전역 변수 설정
global_token = None
global_kakao_id = None

# input 이미지 전처리
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = preprocess(img)
    img = img.unsqueeze(0)
    return img.to(device)


def predict_face_shape(image_path):
    img = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    predicted_class = class_labels[predicted.item()]
    return predicted_class


def predict_least_likely_class(image_path):
    img = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(img)
    probabilities = torch.softmax(outputs, dim=1)
    least_likely_class_prob, least_likely_class_idx = torch.min(probabilities, dim=1)
    least_likely_class_label = class_labels[least_likely_class_idx.item()]
    return least_likely_class_label



# 서버측으로 얼굴형 판별 결과 전송
def post_json_data(predicted_class, least_likely_class_label, global_token):
    url = os.getenv('AWS_URL')

    headers = {
        'Content-Type': 'application/json',
        'Authorization': global_token
    }

    face = {
        "faceShapeBest": predicted_class,
        "faceShapeWorst": least_likely_class_label
    }

    data = json.dumps(face)

    try:
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            print("전송 성공")
        else:
            print(" 전송 실패 Status code:", response.status_code)
            print("오류 내용:", response.text)
    except Exception as e:
        print("에러 발생:", e)

@app.route('/')
def upload_file():
    return render_template('upload4.html')


load_dotenv()

aws_url = os.environ.get("AWS_URL")
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
region_name = os.environ.get('AWS_REGION')
s3_bucket_name = os.environ.get('S3_BUCKET_NAME')



def upload_to_s3(bucket_name, object_key, file_path, aws_access_key, aws_secret_key, region):
    try:
        s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key,
                          region_name=region)
        s3.upload_file(file_path, bucket_name, object_key)
        print("이미지 업로드 성공!")
    except NoCredentialsError:
        print("AWS 인증 정보를 찾을 수 없습니다.")
    except Exception as e:
        print(f"이미지 업로드 실패. 오류: {e}")


def download_images_from_s3(bucket_name, prefix, aws_access_key_id, aws_secret_access_key, region):
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key,
                      region_name=region)
    images = []

    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        for obj in response.get('Contents', []):
            object_key = obj['Key']
            if object_key.endswith(('.jpg', '.png', '.jpeg')):
                img_object = s3.get_object(Bucket=bucket_name, Key=object_key)
                img_data = img_object['Body'].read()
                images.append(img_data)
    except Exception as e:
        print(f"Failed to download images from S3: {e}")

    return images


def generate_and_upload_synthesized_images(file_path, predicted_class, least_likely_class_label, result_dir,
                                           current_datetime, kakao_id):

    # 카카오톡 아이디로 폴더 만들기
    s3_results_dir = f"results/{kakao_id}"
    os.makedirs(result_dir, exist_ok=True)

    # 결과 이미지 생성
    # (생성된 이미지들의 경로를 저장할 리스트 생성)
    generated_images = []

    # S3에 업로드할 결과 이미지의 키 생성
    def generate_image_key(image_name):
        return f"{s3_results_dir}/{image_name}"

    # 결과 이미지 생성 및 업로드
    for ref_image_dir in [f'asset/ref/{predicted_class}', f'asset/ref/{least_likely_class_label}']:
        num_generated_images = 0
        for ref_image_name in os.listdir(ref_image_dir):
            if num_generated_images >= 3:
                break

            target_image_path = os.path.join(ref_image_dir, ref_image_name)

            if not ref_image_name.endswith(('_back.npy', '_aligned.png')):
                synthesized_image_path = generate_synthesized_image(target_image_path, file_path, result_dir)
                generated_image_key = generate_image_key(ref_image_name)
                upload_to_s3(s3_bucket_name, generated_image_key, synthesized_image_path, aws_access_key_id,
                             aws_secret_access_key, region_name)
                generated_images.append(f"https://{s3_bucket_name}.s3.amazonaws.com/{generated_image_key}")
                num_generated_images += 1

    return generated_images


def generate_synthesized_image(target_image_path, source_image_path, output_dir):
    cmd = f"python image_test.py --target_img_path {target_image_path} --source_img_path {source_image_path} --output_dir {output_dir} --use_gpu True"
    subprocess.run(cmd, shell=True, check=True)

    # Assuming the synthesized image is saved with a certain prefix, adjust this part accordingly
    synthesized_image_name = "result_" + os.path.basename(target_image_path)
    synthesized_image_path = os.path.join(output_dir, synthesized_image_name)
    return synthesized_image_path


@app.route('/predict', methods=['POST'])
def predict():

    global global_token
    global global_kakao_id

    file = request.files['file']

    if not global_token or not global_kakao_id:
        return jsonify({'error': 'Missing token or kakao_id'}), 400

    current_dir = os.path.dirname(os.path.abspath(__file__))

    local_file_path = os.path.join(current_dir, 'uploads', file.filename)
    file.save(local_file_path)

    predicted_class = predict_face_shape(local_file_path)
    least_likely_class_label = predict_least_likely_class(local_file_path)

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = os.path.join('results', current_datetime)
    os.makedirs(result_dir, exist_ok=True)

    # Generate and upload synthesized images for both predicted class and least likely class
    generated_images = generate_and_upload_synthesized_images(local_file_path, predicted_class,
                                                              least_likely_class_label, result_dir, current_datetime, global_kakao_id)

    post_json_data(predicted_class, least_likely_class_label, global_token)



    redirect_url = f"https://morak-morak-demo.vercel.app/user?bestFace={predicted_class}&worstFace={least_likely_class_label}"


    return redirect(redirect_url)


@app.route('/api/receivekakaoid', methods=['POST'])
def receive_kakao_id():
    global global_token
    global global_kakao_id

    data = request.json
    global_token = data.get('token')
    global_kakao_id = data.get('kakao_id')

    if not global_token or not global_kakao_id:
        return jsonify({'error': 'Missing token or kakao_id'}), 400

    return jsonify({'message': 'Success', 'kakao_id': global_kakao_id, 'token': global_token}), 200


@app.route('/result')
def result():
    predicted_class = request.args.get('result')
    least_likely_class_label = request.args.get('least_likely_result')
    return render_template('upload4.html', result=predicted_class, least_likely_result=least_likely_class_label)

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000)
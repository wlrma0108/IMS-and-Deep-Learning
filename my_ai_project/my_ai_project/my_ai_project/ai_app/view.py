import torch
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
from io import BytesIO
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# 1. 서버 시작 시, VGG16 로드 (pretrained=True 이면 자동으로 ImageNet 학습된 가중치 사용)
vgg16 = models.vgg16(pretrained=True)
vgg16.eval()  # 평가 모드

# 2. 이미지 전처리에 사용할 변환(transform)
#    - VGG 계열 모델은 224x224, ImageNet 통계로 normalize 필요
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet 통계값
        std=[0.229, 0.224, 0.225]
    )
])

# ImageNet 클래스 라벨(1,000개)를 매핑하기 위해서는 별도의 레이블 파일이 필요.
# 여기서는 예시로 간단히 top-5 확률과 인덱스만 보여줍니다.
# 실제로는 ImageNet 라벨 파일을 불러오거나, custom 클래스면 직접 라벨 매핑을 준비해야 합니다.

@csrf_exempt  # (데모용) 실제 배포 시 csrf_protect + {% csrf_token %} 사용
def classify_image(request):
    if request.method == 'POST':
        # 3. 업로드된 이미지 파일 가져오기
        upload_file = request.FILES.get('image')
        if not upload_file:
            return JsonResponse({'error': '이미지 파일이 필요합니다.'}, status=400)

        try:
            # 4. PIL Image로 열기
            image = Image.open(BytesIO(upload_file.read()))
            # 5. 전처리 수행
            img_t = transform(image)
            # 6. 배치 차원 추가 (N, C, H, W) => (1, C, H, W)
            img_t = img_t.unsqueeze(0)

            # 7. 모델 예측 (shape: [1, 1000])
            with torch.no_grad():
                outputs = vgg16(img_t)

            # 8. 결과 해석 - 가장 확률이 높은 top-5 인덱스 찾기
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top5_prob, top5_catid = torch.topk(probabilities, 5)

            # (선택) ImageNet 레이블이 있는 딕셔너리를 사용하는 경우
            # labels = {0: 'tench', 1: 'goldfish', ..., 999: 'toilet tissue'}
            # top5_labels = [labels[catid.item()] for catid in top5_catid]

            # 여기서는 라벨 매핑 없이 index와 확률만 반환 (실제로는 라벨파일 필요)
            results = []
            for i in range(top5_prob.size(0)):
                cat_id = top5_catid[i].item()
                prob = top5_prob[i].item()
                results.append({
                    'class_index': cat_id,
                    'probability': float(prob)
                })

            return JsonResponse({'results': results})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    else:
        # GET 요청이면 간단한 업로드 폼 페이지 렌더링
        return render(request, 'ai_app/index.html')
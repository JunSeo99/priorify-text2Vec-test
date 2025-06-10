import pandas as pd
import random
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)

from src.config import CATEGORIES_DEFINITIONS, DATA_PATH

def create_ner_effective_samples():
    """NER 일반화가 효과적인 샘플들을 생성"""
    
    # 인명 리스트 (더 많이 추가)
    names = ["김철수", "이영희", "박민수", "최지혜", "정수현", "조영수", "신미나", "윤서준", "한소희", "임태현",
             "김민지", "박서연", "이도현", "정우진", "최유나", "강민호", "송지아", "류준호", "오하은", "장시우"]
    
    # 장소명 리스트 (더 많이 추가)
    places = ["강남역", "홍대입구", "명동", "신촌", "이태원", "압구정", "잠실", "건대입구", "신림", "서울역",
              "종로", "동대문", "을지로", "청담", "논현", "역삼", "선릉", "삼성", "가로수길", "성수동"]
    
    # 날짜 표현 (더 많이 추가)
    dates = ["내일", "모레", "다음주 월요일", "이번 주말", "12월 25일", "1월 1일", "다음달",
             "오는 금요일", "이번주 토요일", "다음주 화요일", "월말", "2월 14일", "3월 5일", "다음주"]
    
    # 시간 표현 (더 많이 추가)
    times = ["오후 2시", "저녁 7시", "오전 10시", "점심시간", "새벽 6시", "밤 11시",
             "오후 3시", "저녁 8시", "오전 9시", "오후 6시", "정오", "오후 1시"]
    
    augmented_data = []
    
    # 친목 카테고리 - 인명이 중요한 경우 (대폭 증가)
    friendship_templates = [
        "{name}와 저녁 약속",
        "{name}와 {place}에서 만나기", 
        "{name}와 {time}에 커피 마시기",
        "{name} 생일파티 참석",
        "{name}와 함께 영화 보기",
        "{name}와 {place}에서 술약속",
        "{name}와 점심 식사",
        "{name}와 {date} {time}에 만나기",
        "{name}와 카페에서 수다떨기",
        "{name}와 함께 쇼핑하기",
        "{name} 결혼식 참석",
        "{name}와 볼링장 가기",
        "{name}와 노래방 가기",
        "{name}와 게임하기",
        "{name}와 운동하기"
    ]
    
    for template in friendship_templates:
        for _ in range(5):  # 3개에서 5개로 증가
            sample = template.format(
                name=random.choice(names),
                place=random.choice(places), 
                time=random.choice(times),
                date=random.choice(dates)
            )
            augmented_data.append({"title": sample, "categories": "친목"})
    
    # 연애 카테고리 - 인명이 중요한 경우 (대폭 증가)
    romance_templates = [
        "{name}와 데이트하기",
        "{name}와 {place}에서 데이트",
        "{name}와 기념일 챙기기", 
        "{name}에게 선물하기",
        "{name}와 {date} {time}에 만나기",
        "{name}와 로맨틱한 저녁식사",
        "{name}와 함께 여행가기",
        "{name}와 영화 데이트",
        "{name}와 산책하기",
        "{name}와 커플링 맞추기"
    ]
    
    for template in romance_templates:
        for _ in range(4):  # 2개에서 4개로 증가
            sample = template.format(
                name=random.choice(names),
                place=random.choice(places),
                time=random.choice(times), 
                date=random.choice(dates)
            )
            augmented_data.append({"title": sample, "categories": "연애"})
    
    # 예약 카테고리 - 장소가 중요한 경우 (증가)
    reservation_templates = [
        "{place}에 있는 식당 예약",
        "{place} 맛집 예약하기",
        "{place} 카페 예약",
        "{place} 병원 예약하기",
        "{place} 미용실 예약",
        "{place} 헬스장 등록",
        "{place} 학원 상담 예약",
        "{place} 치과 예약",
        "{place} 피부과 예약",
        "{place} 마사지샵 예약"
    ]
    
    for template in reservation_templates:
        for _ in range(3):  # 2개에서 3개로 증가
            sample = template.format(place=random.choice(places))
            augmented_data.append({"title": sample, "categories": "예약"})
    
    # 여행 카테고리 - 장소가 중요한 경우 (증가)
    travel_templates = [
        "{place}로 여행 계획",
        "{place} 여행 준비하기", 
        "{place} 숙소 예약",
        "{place} 관광지 알아보기",
        "{place} 맛집 찾아보기",
        "{place} 여행 일정 짜기",
        "{place} 항공권 예약",
        "{place} 렌터카 예약"
    ]
    
    travel_places = ["부산", "제주도", "경주", "전주", "강릉", "속초", "여수", "대구", "인천", "수원",
                     "통영", "안동", "춘천", "포항", "울산", "창원", "진주", "목포", "순천", "광주"]
    for template in travel_templates:
        for _ in range(3):  # 2개에서 3개로 증가
            sample = template.format(place=random.choice(travel_places))
            augmented_data.append({"title": sample, "categories": "여행"})
    
    # 통화 카테고리 - 인명이 중요한 경우 (증가)
    call_templates = [
        "{name}와 통화하기",
        "{name}에게 전화하기",
        "{name}와 {time}에 전화 약속", 
        "{name}와 영상통화",
        "{name}에게 안부 전화",
        "{name}와 업무 통화",
        "{name}와 화상회의",
        "{name}와 긴급 통화"
    ]
    
    for template in call_templates:
        for _ in range(3):  # 2개에서 3개로 증가
            sample = template.format(
                name=random.choice(names),
                time=random.choice(times)
            )
            augmented_data.append({"title": sample, "categories": "통화"})
    
    # 치료 카테고리 - 장소가 중요한 경우 (증가)
    treatment_templates = [
        "{place}에 있는 병원 방문",
        "{place} 치과 예약",
        "{place} 한의원 방문",
        "{place} 정형외과 치료",
        "{place} 피부과 시술",
        "{place} 성형외과 상담",
        "{place} 내과 진료",
        "{place} 안과 검진"
    ]
    
    for template in treatment_templates:
        for _ in range(3):  # 2개에서 3개로 증가
            sample = template.format(place=random.choice(places))
            augmented_data.append({"title": sample, "categories": "치료"})
    
    # 가족 카테고리 추가 (인명이 중요)
    family_templates = [
        "{name} 어머니 생신 챙기기",
        "{name} 아버지와 식사하기",
        "{name} 할머니 댁 방문",
        "{name}와 가족 모임",
        "{name} 형제와 만나기",
        "{name} 사촌과 약속"
    ]
    
    for template in family_templates:
        for _ in range(3):
            sample = template.format(name=random.choice(names))
            augmented_data.append({"title": sample, "categories": "가족"})
    
    # 업무 카테고리 추가 (인명과 장소 중요)
    work_templates = [
        "{name}와 {place}에서 회의",
        "{name} 팀장과 미팅",
        "{place} 사무실에서 프레젠테이션",
        "{name}와 업무 협의",
        "{place}에서 고객 미팅"
    ]
    
    for template in work_templates:
        for _ in range(3):
            sample = template.format(
                name=random.choice(names),
                place=random.choice(places)
            )
            augmented_data.append({"title": sample, "categories": "업무"})
    
    return augmented_data

def augment_dataset():
    """기존 데이터셋에 NER 효과적인 샘플들을 추가"""
    
    # 기존 데이터 로드
    existing_df = pd.read_csv(DATA_PATH)
    print(f"기존 데이터 개수: {len(existing_df)}")
    
    # 새로운 샘플 생성
    augmented_samples = create_ner_effective_samples()
    augmented_df = pd.DataFrame(augmented_samples)
    print(f"추가할 샘플 개수: {len(augmented_df)}")
    
    # 데이터 결합
    combined_df = pd.concat([existing_df, augmented_df], ignore_index=True)
    
    # 중복 제거 (title 기준)
    combined_df = combined_df.drop_duplicates(subset=['title'], keep='first')
    print(f"중복 제거 후 전체 데이터 개수: {len(combined_df)}")
    
    # 새로운 파일로 저장
    output_path = DATA_PATH.replace('.csv', '_augmented.csv')
    combined_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"증강된 데이터셋 저장: {output_path}")
    
    return output_path

if __name__ == "__main__":
    augmented_path = augment_dataset()
    print("\n=== 증강 완료 ===")
    print("이제 다음 명령어로 성능을 비교해보세요:")
    print(f"python src/scripts/compare_versions.py --data_path {augmented_path}") 
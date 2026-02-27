import streamlit as st
import pandas as pd
import json
import re
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
import os

# --- Constants & Configuration ---
# JSON(만원) / 10 = HTML(만 단위 표기)
PRICE_CONVERSION_FACTOR = 10 

# --- Utility Functions ---

def parse_html_price(text):
    """HTML의 '170만', '4,500만' 등을 숫자(만원)로 변환"""
    if not text:
        return 0
    # 쉼표 제거
    text = text.replace(',', '')
    # '만' 제거하고 숫자만 추출
    match = re.search(r'([\d\.]+)', text)
    if match:
        val = float(match.group(1))
        return int(val * 1) # 이미 '만' 단위이므로 그대로 숫자로 간주 (만원 기준)
    return 0

def format_price(value_man, unit_type):
    """만원 단위 숫자를 만 단위(String) 또는 원 단위(String)로 변환"""
    if unit_type == "만 단위 표기":
        return f"{value_man:,}만"
    else:
        return f"{value_man * 10000:,}원"

import sqlite3

class DataEngine:
    def __init__(self, md_path, db_path):
        self.md_path = md_path
        self.db_path = db_path
        self.raw_content = ""
        self.df = pd.DataFrame()
        self.html_data = {}

    @st.cache_data
    def load_full_data(_self):
        """SQLite에서 전체 데이터를 로드하고 캐싱"""
        if not os.path.exists(_self.db_path):
            return pd.DataFrame()
        try:
            conn = sqlite3.connect(_self.db_path)
            # 'items' 테이블에서 전체 데이터 로드
            df = pd.read_sql_query("SELECT * FROM items", conn)
            conn.close()
            return df
        except Exception as e:
            st.error(f"DB 로드 실패: {e}")
            return pd.DataFrame()

    def load_and_parse_sample(self):
        """마크다운 샘플 데이터 파싱 (기존 로직 유지)"""
        if not os.path.exists(self.md_path):
            return False, f"파일을 찾을 수 없습니다: {self.md_path}"

        with open(self.md_path, "r", encoding="utf-8") as f:
            self.raw_content = f.read()

        # 1. JSON 추출 로직 (불완전한 JSON 대응)
        try:
            content = self.raw_content
            marker = "위 정보"
            if marker in content:
                json_source = content.split(marker)[0].strip()
            else:
                json_source = content.strip()

            json_source = json_source.rstrip(',')
            open_sq, close_sq = json_source.count('['), json_source.count(']')
            if open_sq > close_sq: json_source += ']' * (open_sq - close_sq)
            open_br, close_br = json_source.count('{'), json_source.count('}')
            if open_br > close_br: json_source += '}' * (open_br - close_br)

            try:
                data = json.loads(json_source)
                self.sample_items = data.get("items", [])
            except json.JSONDecodeError:
                item_matches = re.findall(r'(\{.+\})', json_source, re.DOTALL)
                self.sample_items = []
                for m in item_matches:
                    try:
                        obj = json.loads(m)
                        if isinstance(obj, dict): self.sample_items.append(obj)
                    except: continue
        except Exception as e:
            return False, f"JSON 처리 중 오류 발생: {e}"

        # 2. HTML 추출 및 파싱
        soup = BeautifulSoup(self.raw_content, 'html.parser')
        price_table = soup.find('div', class_='price-container')
        if price_table:
            rows = price_table.find_all('tr')
            for row in rows:
                th = row.find('th').get_text(strip=True) if row.find('th') else ""
                td = row.find('td').get_text(strip=True) if row.find('td') else ""
                if '월세' in th: self.html_data['monthlyRent'] = td
                elif '보증금' in th: self.html_data['deposit'] = td
                elif '권리금' in th: self.html_data['premium'] = td
                elif '관리비' in th: self.html_data['maintenanceFee'] = td

        return True, "성공"

# 경로 설정
DATA_PATH = "nemostore/data/data_json_html.md"
DB_PATH = "nemostore/data/nemo_products.db"

engine = DataEngine(DATA_PATH, DB_PATH)
full_df = engine.load_full_data()
sample_success, sample_msg = engine.load_and_parse_sample()

# 사이드바 설정
st.sidebar.title("🏢 Nemostore Admin")
nav = st.sidebar.radio("Navigation", ["Overview", "Listings Table", "Price Analytics", "Detail & Validation"])
unit_toggle = st.sidebar.selectbox("금액 단위 설정", ["만 단위 표기", "원 단위 표기"])

if full_df.empty:
    st.warning("데이터베이스가 비어있거나 찾을 수 없습니다. 샘플 데이터로 대체합니다.")
    # 샘플 데이터가 파싱 성공했을 경우 그 데이터를 사용
    df = pd.DataFrame(engine.sample_items) if sample_success else pd.DataFrame()
else:
    df = full_df

if df.empty:
    st.error("사용 가능한 데이터가 없습니다.")
    st.stop()

# --- Page Logic ---

if nav == "Overview":
    st.header("📊 Market Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("총 매물 수", f"{len(df):,}개")
    with col2: st.metric("평균 월세", format_price(df['monthlyRent'].mean() / PRICE_CONVERSION_FACTOR, unit_toggle))
    with col3: st.metric("평균 보증금", format_price(df['deposit'].mean() / PRICE_CONVERSION_FACTOR, unit_toggle))
    with col4: st.metric("평균 권리금", format_price(df['premium'].mean() / PRICE_CONVERSION_FACTOR, unit_toggle))
    with col5: st.metric("평균 면적", f"{df['size'].mean():.2f} ㎡")
            
    st.subheader("Random Sample Listing")
    sample_item = df.sample(1).iloc[0]
    if 'previewPhotoUrl' in sample_item and sample_item['previewPhotoUrl']:
        st.image(sample_item['previewPhotoUrl'], width=600, caption=sample_item['title'])

elif nav == "Listings Table":
    st.header("📋 Property Listings")
    
    # 필터
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        # 업종 필터 (DB 연동으로 확장된 옵션)
        biz_types = sorted([t for t in df['businessLargeCodeName'].unique() if t and t != '0'])
        biz_filter = st.multiselect("업종 필터", biz_types, default=biz_types[:3] if len(biz_types) > 3 else biz_types)
    
    with col_f2:
        min_rent = int(df['monthlyRent'].min())
        max_rent = int(df['monthlyRent'].max())
        if min_rent == max_rent:
            st.info(f"현재 월세가 단일 값({min_rent}만원)으로 고정되어 있습니다.")
            rent_range = (min_rent, max_rent)
        else:
            rent_range = st.slider("월세 범위 (만원)", min_rent, max_rent, (min_rent, max_rent))

    filtered_df = df[
        (df['businessLargeCodeName'].isin(biz_filter)) &
        (df['monthlyRent'].between(rent_range[0], rent_range[1]))
    ]
    
    st.markdown(f"**검색 결과:** {len(filtered_df)}개")
    st.dataframe(filtered_df, use_container_width=True)

elif nav == "Price Analytics":
    st.header("📈 Price Analytics (Full Dataset)")
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.subheader("Monthly Rent Distribution")
        fig, ax = plt.subplots()
        # 아웃라이어 조정을 위해 상위 5% 제외 후 시각화 (선택사항)
        upper_limit = df['monthlyRent'].quantile(0.95)
        sns.histplot(df[df['monthlyRent'] <= upper_limit]['monthlyRent'], kde=True, ax=ax, color='skyblue')
        ax.set_xlabel("Monthly Rent (10,000 KRW)")
        st.pyplot(fig)
        
    with col_c2:
        st.subheader("Sector-wise Average Rent")
        fig, ax = plt.subplots()
        sector_avg = df.groupby('businessLargeCodeName')['monthlyRent'].mean().sort_values(ascending=False).head(10)
        sector_avg.plot(kind='barh', ax=ax, color='salmon')
        st.pyplot(fig)

elif nav == "Detail & Validation":
    st.header("🔍 Cross-Validation (Sample vs JSON)")
    
    if not sample_success:
        st.error("교차 검증을 위한 샘플 데이터를 불러오지 못했습니다.")
    else:
        sample_df = pd.DataFrame(engine.sample_items)
        selected_idx = st.selectbox("검증할 샘플 매물을 선택하세요", sample_df.index, format_func=lambda x: sample_df.loc[x, 'title'])
        item = sample_df.loc[selected_idx]
        
        st.subheader(f"Verification: {item['title']}")
        
        validation_data = []
        fields = [('monthlyRent', '월세'), ('deposit', '보증금'), ('premium', '권리금'), ('maintenanceFee', '관리비')]
        
        for f_key, f_name in fields:
            json_val_man = item[f_key] / PRICE_CONVERSION_FACTOR
            html_str = engine.html_data.get(f_key, "N/A")
            html_val_man = parse_html_price(html_str)
            status = "✅ OK" if json_val_man == html_val_man else "❌ DIFF"
            diff = json_val_man - html_val_man
            
            validation_data.append({
                "항목": f_name,
                "JSON (만 단위 환산)": f"{json_val_man:,}만",
                "HTML (추출 텍스트)": html_str,
                "HTML (추출 숫자)": f"{html_val_man:,}만",
                "상태": status,
                "차이": f"{diff:,}만" if diff != 0 else "-"
            })
            
        st.table(pd.DataFrame(validation_data))

st.sidebar.markdown("---")
st.sidebar.caption(f"Senior Analyst Dashboard v2.1 | Data: {len(df)} items")

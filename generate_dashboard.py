#!/usr/bin/env python3
"""삼성전자 & SK하이닉스 주식 대시보드 생성기 — 매일 오전 6시(KST) 자동 실행"""

import yfinance as yf
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import sys

KST = pytz.timezone('Asia/Seoul')

STOCKS = {
    'samsung': {'ticker': '005930.KS', 'name': '삼성전자', 'color': '#1976D2', 'color_pred': '#64B5F6'},
    'hynix':   {'ticker': '000660.KS', 'name': 'SK하이닉스', 'color': '#FF6B00', 'color_pred': '#FFB74D'},
}

# ─── 데이터 수집 ────────────────────────────────────────────────────────────────

def fetch_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='14mo')
        info = {}
        try:
            info = stock.info
        except Exception as e:
            print(f"  info 로드 실패 ({ticker}): {e}", file=sys.stderr)
        return hist, info
    except Exception as e:
        print(f"  데이터 로드 실패 ({ticker}): {e}", file=sys.stderr)
        return pd.DataFrame(), {}

# ─── 기술적 지표 ─────────────────────────────────────────────────────────────────

def calc_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=window).mean()
    rs = gain / loss.where(loss != 0, other=np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig

def calc_bb(prices, window=20, num_std=2):
    sma = prices.rolling(window=window, min_periods=window).mean()
    std = prices.rolling(window=window, min_periods=window).std()
    return sma + num_std * std, sma, sma - num_std * std

def predict_prices(hist, info, periods=252):
    close = hist['Close']
    analyst_target = info.get('targetMeanPrice')
    analyst_high   = info.get('targetHighPrice')
    analyst_low    = info.get('targetLowPrice')
    recent = close.tail(120)
    x = np.arange(len(recent))
    coeffs = np.polyfit(x, recent.values, 1)
    future_x = np.arange(len(recent), len(recent) + periods)
    trend = np.polyval(coeffs, future_x)
    if analyst_target and analyst_target > 0:
        blend = np.linspace(0, 1, periods)
        predicted = trend * (1 - blend) + float(analyst_target) * blend
    else:
        predicted = trend
    last_date = hist.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=periods)
    return future_dates, predicted, analyst_target, analyst_high, analyst_low

def get_signal(hist_full, hist_1y):
    close_1y   = hist_1y['Close']
    close_full = hist_full['Close']
    price = close_1y.iloc[-1]
    rsi_val = calc_rsi(close_1y).iloc[-1]
    macd_line, macd_sig, _ = calc_macd(close_1y)
    sma20  = close_1y.rolling(20).mean().iloc[-1]
    sma50  = close_1y.rolling(50).mean().iloc[-1]
    sma200 = close_full.rolling(200).mean().iloc[-1]
    bb_upper, _, bb_lower = calc_bb(close_1y)
    score, details = 0, []

    if not np.isnan(rsi_val):
        if rsi_val < 30:
            score += 2; details.append({'label': f'RSI 과매도 ({rsi_val:.1f})', 'type': 'buy'})
        elif rsi_val > 70:
            score -= 2; details.append({'label': f'RSI 과매수 ({rsi_val:.1f})', 'type': 'sell'})
        else:
            details.append({'label': f'RSI 중립 ({rsi_val:.1f})', 'type': 'neutral'})

    if macd_line.iloc[-1] > macd_sig.iloc[-1]:
        score += 1; details.append({'label': 'MACD 골든크로스', 'type': 'buy'})
    else:
        score -= 1; details.append({'label': 'MACD 데드크로스', 'type': 'sell'})

    for d, v, lbl in [(price > sma20, sma20, '20일선'), (price > sma50, sma50, '50일선')]:
        if d:
            score += 1; details.append({'label': f'{lbl} 위 ({v:,.0f}원)', 'type': 'buy'})
        else:
            score -= 1; details.append({'label': f'{lbl} 아래 ({v:,.0f}원)', 'type': 'sell'})

    if not np.isnan(sma200):
        if price > sma200:
            score += 1; details.append({'label': f'200일선 위 ({sma200:,.0f}원)', 'type': 'buy'})
        else:
            score -= 1; details.append({'label': f'200일선 아래 ({sma200:,.0f}원)', 'type': 'sell'})

    bbu, bbl = bb_upper.iloc[-1], bb_lower.iloc[-1]
    if not np.isnan(bbl) and price <= bbl:
        score += 1; details.append({'label': '볼린저 하단 지지', 'type': 'buy'})
    elif not np.isnan(bbu) and price >= bbu:
        score -= 1; details.append({'label': '볼린저 상단 저항', 'type': 'sell'})

    if score >= 4:   overall, stype = '강력 매수', 'strong-buy'
    elif score >= 2: overall, stype = '매수',     'buy'
    elif score <= -4:overall, stype = '강력 매도', 'strong-sell'
    elif score <= -2:overall, stype = '매도',     'sell'
    else:            overall, stype = '중립',     'neutral'

    def sf(v): return round(float(v), 1) if not np.isnan(v) else None
    return {
        'overall': overall, 'type': stype, 'score': score, 'details': details,
        'rsi': sf(rsi_val), 'sma20': round(float(sma20), 0), 'sma50': round(float(sma50), 0),
        'sma200': round(float(sma200), 0) if not np.isnan(sma200) else None,
        'macd': round(float(macd_line.iloc[-1]), 2), 'macd_signal': round(float(macd_sig.iloc[-1]), 2),
        'bb_upper': round(float(bbu), 0) if not np.isnan(bbu) else None,
        'bb_lower': round(float(bbl), 0) if not np.isnan(bbl) else None,
    }

# ─── 종목 처리 ────────────────────────────────────────────────────────────────

def clean_series(s):
    return [round(float(v), 2) if (v is not None and not np.isnan(v)) else None for v in s]

def clean_prices(s):
    return [round(float(v), 0) if (v is not None and not np.isnan(v)) else None for v in s]

def process_stock(key):
    cfg = STOCKS[key]
    print(f"  [{cfg['name']}] 데이터 로딩 중...")
    hist_full, info = fetch_data(cfg['ticker'])
    if hist_full.empty:
        return None

    one_year_ago = hist_full.index[-1] - pd.DateOffset(years=1)
    hist_1y = hist_full[hist_full.index >= one_year_ago]
    close = hist_1y['Close']
    price = float(close.iloc[-1])
    prev  = float(close.iloc[-2]) if len(close) > 1 else price

    sma20_s    = close.rolling(20).mean()
    sma50_s    = close.rolling(50).mean()
    sma200_1y  = hist_full['Close'].rolling(200).mean()[hist_full.index >= one_year_ago]
    bb_upper, _, bb_lower = calc_bb(close)
    rsi_s   = calc_rsi(close)
    macd_l, macd_sig_s, macd_h = calc_macd(close)

    future_dates, predicted, a_target, a_high, a_low = predict_prices(hist_full, info)
    signal = get_signal(hist_full, hist_1y)

    returns    = close.pct_change().dropna()
    volatility = float(returns.tail(30).std() * np.sqrt(252) * 100)
    ytd_return = float((price - float(close.iloc[0])) / float(close.iloc[0]) * 100)

    def sf(v):
        if v is None: return None
        try:
            f = float(v)
            return None if np.isnan(f) else f
        except Exception:
            return None

    hist_len = len(hist_1y)
    pred_len  = len(future_dates)
    all_dates = [d.strftime('%Y-%m-%d') for d in hist_1y.index] + \
                [d.strftime('%Y-%m-%d') for d in future_dates]

    return {
        'key': key, 'name': cfg['name'], 'ticker': cfg['ticker'],
        'color': cfg['color'], 'color_pred': cfg['color_pred'],
        'price': round(price, 0), 'change': round(price - prev, 0),
        'change_pct': round((price - prev) / prev * 100, 2),
        'high_52w': round(float(close.max()), 0), 'low_52w': round(float(close.min()), 0),
        'volume': int(hist_1y['Volume'].iloc[-1]), 'avg_volume': int(hist_1y['Volume'].mean()),
        'volatility': round(volatility, 1), 'ytd_return': round(ytd_return, 1),
        'pe': sf(info.get('trailingPE')), 'pb': sf(info.get('priceToBook')),
        'eps': sf(info.get('trailingEps')), 'market_cap': sf(info.get('marketCap')),
        'dividend_yield': sf(info.get('dividendYield')),
        'analyst_target': sf(a_target), 'analyst_high': sf(a_high), 'analyst_low': sf(a_low),
        'analyst_count': info.get('numberOfAnalystOpinions'),
        'recommendation': info.get('recommendationKey'),
        'signal': signal,
        'chart': {
            'dates': all_dates, 'hist_len': hist_len,
            'prices': clean_prices(close.tolist()) + [None] * pred_len,
            'pred':   [None] * (hist_len - 1) + [round(price, 0)] + [round(float(v), 0) for v in predicted],
            'sma20':  clean_prices(sma20_s.tolist()) + [None] * pred_len,
            'sma50':  clean_prices(sma50_s.tolist()) + [None] * pred_len,
            'sma200': clean_prices(sma200_1y.tolist()) + [None] * pred_len,
            'bb_upper': clean_prices(bb_upper.tolist()) + [None] * pred_len,
            'bb_lower': clean_prices(bb_lower.tolist()) + [None] * pred_len,
            'volume':      [int(v) for v in hist_1y['Volume'].tolist()] + [None] * pred_len,
            'rsi':         clean_series(rsi_s.tolist()) + [None] * pred_len,
            'macd':        clean_series(macd_l.tolist()) + [None] * pred_len,
            'macd_signal': clean_series(macd_sig_s.tolist()) + [None] * pred_len,
            'macd_hist':   clean_series(macd_h.tolist()) + [None] * pred_len,
        },
    }

# ─── HTML 조각 빌더 ──────────────────────────────────────────────────────────────

def strength_bar(name, score, desc):
    color = '#10b981' if score >= 80 else '#f59e0b' if score >= 60 else '#ef4444'
    return f"""
    <div class="sfactor">
      <div class="sf-row">
        <span class="sf-name">{name}</span>
        <span class="sf-score" style="color:{color}">{score}</span>
      </div>
      <div class="sf-track"><div class="sf-fill" style="width:{score}%;background:{color}"></div></div>
      <div class="sf-desc">{desc}</div>
    </div>"""

def risk_item(name, level, desc):
    lc = {'높음': ('#ef4444', '#ef444420'), '중간': ('#f59e0b', '#f59e0b20'), '낮음': ('#10b981', '#10b98120')}
    fg, bg = lc.get(level, ('#9ca3af', '#9ca3af20'))
    return f"""
    <div class="rfactor">
      <div class="rf-row">
        <span class="rf-name">{name}</span>
        <span class="rf-badge" style="color:{fg};background:{bg};border:1px solid {fg}40">{level}</span>
      </div>
      <div class="rf-desc">{desc}</div>
    </div>"""

def scenario_col(label, price, current, prob, color, bg, items):
    pct  = (price - current) / current * 100
    sign = '+' if pct >= 0 else ''
    return f"""
    <div class="scen-col" style="border-top:3px solid {color}">
      <div class="scen-label" style="color:{color}">{label}</div>
      <div class="scen-prob" style="color:{color}">확률 {prob}%</div>
      <div class="scen-price">{price:,.0f}원</div>
      <div class="scen-pct" style="color:{color}">{sign}{pct:.1f}%</div>
      <ul class="scen-list">{''.join(f'<li>{i}</li>' for i in items)}</ul>
    </div>"""

# ─── 심층 분석 섹션 ─────────────────────────────────────────────────────────────

def deep_analysis_section(sam, hyx):
    sp = sam['price']; hp = hyx['price']

    # 삼성 강점
    sam_strengths = (
        strength_bar('종합 반도체 포트폴리오', 94, 'DRAM·NAND·파운드리·시스템반도체 4개 축 보유. 단일 사업 리스크 분산') +
        strength_bar('DRAM 시장점유율 1위', 88, '글로벌 DRAM 시장 약 43% 점유. 삼성 브랜드로 프리미엄 고객 다수 확보') +
        strength_bar('재무 건전성', 92, '순현금 보유 규모 업계 최고 수준. AA+ 신용등급, 경기 침체에도 투자 여력 충분') +
        strength_bar('글로벌 브랜드 & 고객 기반', 89, '스마트폰·가전 글로벌 1위. 애플·구글·마이크로소프트 등 핵심 B2B 고객') +
        strength_bar('배당 안정성', 85, '연 2~3% 배당수익률 꾸준히 지급. 특별배당 포함 주주환원 강화 추세') +
        strength_bar('HBM3E 생산 능력', 63, '2024년 하반기 양산 진입. SK하이닉스 대비 약 6개월 뒤처지나 격차 축소 중') +
        strength_bar('파운드리 GAA 공정', 58, '3nm GAA 세계 최초 양산. 수율 개선 중이나 TSMC 대비 격차 여전히 존재')
    )

    # 삼성 리스크
    sam_risks = (
        risk_item('HBM 시장 열위 장기화', '높음', '엔비디아 HBM3E 퀄 검증 반복 지연. 하이닉스·마이크론에 점유율 추가 손실 가능') +
        risk_item('미·중 반도체 수출 규제', '높음', '미국의 HBM 포함 고급 메모리 대중 수출 규제 강화 검토 중. 중국 매출 비중 약 15%') +
        risk_item('파운드리 수율·고객 이탈', '높음', '3nm 수율 50% 초반 추정. TSMC에서 복귀 기대 고객 확보 부진') +
        risk_item('중국 CXMT 등 추격', '중간', 'DDR5 양산 진입. 저가 시장 잠식 가속화. 2026년 이후 DRAM 가격 압박 현실화 우려') +
        risk_item('사업부 간 이익 편차', '중간', '메모리 흑자·파운드리 적자 병행. 자원 배분 갈등 및 전략 집중력 분산 리스크') +
        risk_item('스마트폰 시장 성숙', '중간', '글로벌 스마트폰 출하 정체. 갤럭시 AI폰 전환 효과 아직 미미')
    )

    # 하이닉스 강점
    hyx_strengths = (
        strength_bar('HBM 글로벌 시장점유율 1위', 96, '고대역폭 메모리(HBM) 시장 약 50% 점유. NVIDIA 주력 공급사로 독보적 지위') +
        strength_bar('AI 수요 직접 수혜', 95, 'H100·H200·B200 등 NVIDIA 최신 GPU에 HBM 독점적 공급. AI 투자 → 즉시 수혜') +
        strength_bar('전략적 파트너십', 90, 'NVIDIA·AMD·Intel·구글 TPU 공급. 단일 고객 의존도 낮추는 다변화 진행 중') +
        strength_bar('DRAM 기술 경쟁력', 91, '1a nm 공정 양산, 1b nm 전환 진행. 최선단 공정에서 삼성과 대등하거나 앞서는 영역') +
        strength_bar('HBM4 선행 개발', 85, '2025년 하반기 HBM4 양산 목표. 경쟁사보다 12~18개월 앞선 로드맵 유지') +
        strength_bar('수익성 레버리지', 78, '메모리 사이클 상승기 영업이익 폭발적 성장. 2024년 연간 영업이익 수십 조 회복') +
        strength_bar('순수 메모리 집중', 75, '비핵심 사업 없이 메모리에만 집중. 기술 투자 효율성·의사결정 속도 우위')
    )

    # 하이닉스 리스크
    hyx_risks = (
        risk_item('NVIDIA 의존도', '높음', 'HBM 매출의 70% 이상이 NVIDIA 향. 엔비디아 수요 감소·공급업체 다변화 시 직격탄') +
        risk_item('메모리 사이클 레버리지', '높음', '순수 메모리 기업 특성상 다운사이클에서 이익 급감. 2022~23년 영업손실 재현 가능성') +
        risk_item('AI 투자 버블 우려', '높음', '빅테크 AI Capex 과잉 투자 피로감 확산. 수요 급냉 시 HBM 재고 급증·가격 급락 위험') +
        risk_item('중국 우시 공장 리스크', '중간', '우시 DRAM 공장이 전체 생산량의 약 40%. 미중 갈등 심화 시 수출규제 대상 편입 가능') +
        risk_item('NAND 사업 수익성', '중간', 'NAND 부문 가격 회복 더디며 지속 적자. 솔리다임(인텔 SSD 사업 인수) 통합 효과 아직 미미') +
        risk_item('삼성전자 HBM 추격', '중간', '삼성이 2025년 HBM3E 퀄 통과 시 점유율 10~15%p 반환 압력. ASP 하락 동반 가능') +
        risk_item('고부채 투자 구조', '중간', 'HBM·DRAM 대규모 CAPEX로 부채 증가. 금리 상승기 이자 부담 확대')
    )

    return f"""
  <!-- 심층 강점 & 리스크 분석 -->
  <div class="section-card">
    <div class="section-head">
      <div class="section-title">🔍 종목별 강점 &amp; 리스크 심층 분석</div>
      <div class="tabs">
        <button class="tab on"  onclick="switchAnalysis('samsung', this)">삼성전자</button>
        <button class="tab off" onclick="switchAnalysis('hynix', this)">SK하이닉스</button>
      </div>
    </div>

    <!-- 삼성전자 -->
    <div id="anal-samsung" class="anal-pane">
      <div class="str-risk-grid">
        <div>
          <div class="pane-title">💪 핵심 강점 <span class="pane-sub">(0–100점)</span></div>
          {sam_strengths}
        </div>
        <div>
          <div class="pane-title">⚠️ 주요 리스크</div>
          {sam_risks}
        </div>
      </div>
    </div>

    <!-- SK하이닉스 -->
    <div id="anal-hynix" class="anal-pane" style="display:none">
      <div class="str-risk-grid">
        <div>
          <div class="pane-title">💪 핵심 강점 <span class="pane-sub">(0–100점)</span></div>
          {hyx_strengths}
        </div>
        <div>
          <div class="pane-title">⚠️ 주요 리스크</div>
          {hyx_risks}
        </div>
      </div>
    </div>
  </div>"""


def scenarios_section(sam, hyx):
    sp = sam['price']; hp = hyx['price']

    # 시나리오 가격 계산
    sam_bull  = sam['analyst_high']  or round(sp * 1.45, -2)
    sam_base  = sam['analyst_target'] or round(sp * 1.15, -2)
    sam_bear  = sam['analyst_low']   or round(sp * 0.78, -2)
    hyx_bull  = hyx['analyst_high']  or round(hp * 1.65, -2)
    hyx_base  = hyx['analyst_target'] or round(hp * 1.30, -2)
    hyx_bear  = hyx['analyst_low']   or round(hp * 0.62, -2)

    sam_cols = (
        scenario_col('강세 시나리오 🚀', sam_bull, sp, 25, '#10b981', '#10b98110', [
            'HBM3E 엔비디아 퀄 통과 성공',
            '파운드리 GAA 수율 60%+ 달성',
            'AI 서버 수요 지속 폭증',
            'DRAM 슈퍼사이클 진입',
            '파운드리 고객사 추가 확보',
        ]) +
        scenario_col('기본 시나리오 📊', sam_base, sp, 50, '#3b82f6', '#3b82f610', [
            '메모리 점진적 회복 지속',
            'HBM 점유율 소폭 증가',
            '파운드리 수율 개선 진행',
            '스마트폰·가전 안정 수요',
            '현재 밸류에이션 적정 수준',
        ]) +
        scenario_col('약세 시나리오 📉', sam_bear, sp, 25, '#ef4444', '#ef444410', [
            'HBM 퀄 재차 실패 장기화',
            '메모리 다운사이클 재진입',
            '파운드리 고객 이탈 가속',
            '미·중 갈등에 중국 매출 급감',
            'CXMT 추격으로 DRAM 가격 하락',
        ])
    )

    hyx_cols = (
        scenario_col('강세 시나리오 🚀', hyx_bull, hp, 30, '#10b981', '#10b98110', [
            'AI Capex 폭발적 증가 지속',
            'HBM4 독점적 선점 성공',
            'NVIDIA 외 AMD·구글 비중 확대',
            'NAND 가격 회복 및 흑자 전환',
            '메모리 슈퍼사이클 동반 수혜',
        ]) +
        scenario_col('기본 시나리오 📊', hyx_base, hp, 45, '#3b82f6', '#3b82f610', [
            'AI 서버 수요 안정적 성장',
            'HBM3E 점유율 45~50% 유지',
            'DRAM 가격 완만한 상승 유지',
            'NAND 손익분기 근접',
            '우시 공장 규제 리스크 제한적',
        ]) +
        scenario_col('약세 시나리오 📉', hyx_bear, hp, 25, '#ef4444', '#ef444410', [
            'AI 투자 거품 붕괴·수요 급냉',
            'NVIDIA 공급업체 다변화 가속',
            '메모리 다운사이클 재진입',
            '삼성 HBM 점유율 대폭 반환',
            '우시 공장 규제 적용 현실화',
        ])
    )

    return f"""
  <!-- 투자 시나리오 -->
  <div class="section-card">
    <div class="section-head">
      <div class="section-title">🎯 12개월 투자 시나리오 분석</div>
      <div class="tabs">
        <button class="tab on"  onclick="switchScen('samsung', this)">삼성전자</button>
        <button class="tab off" onclick="switchScen('hynix', this)">SK하이닉스</button>
      </div>
    </div>
    <div class="scen-note">현재가 기준 상대 수익률 · 애널리스트 목표가 및 업황 분석 반영 · 투자 참고용</div>

    <div id="scen-samsung" class="scen-pane">
      <div class="scen-current">현재가 <strong>{sp:,.0f}원</strong></div>
      <div class="scen-grid">{sam_cols}</div>
    </div>
    <div id="scen-hynix" class="scen-pane" style="display:none">
      <div class="scen-current">현재가 <strong>{hp:,.0f}원</strong></div>
      <div class="scen-grid">{hyx_cols}</div>
    </div>
  </div>"""


def industry_section():
    return """
  <!-- 반도체 산업 분석 -->
  <div class="section-card">
    <div class="section-title" style="margin-bottom:20px">🏭 반도체 산업 & HBM 시장 분석</div>

    <div class="industry-grid">

      <div class="ind-block">
        <div class="ind-title">📈 HBM 시장 성장 전망</div>
        <div class="hbm-bar-wrap">
          <div class="hbm-row"><span>2022</span><div class="hbm-bar" style="width:10%"></div><span class="hbm-val">~2조원</span></div>
          <div class="hbm-row"><span>2023</span><div class="hbm-bar" style="width:22%"></div><span class="hbm-val">~5조원</span></div>
          <div class="hbm-row"><span>2024</span><div class="hbm-bar" style="width:48%"></div><span class="hbm-val">~14조원</span></div>
          <div class="hbm-row"><span>2025E</span><div class="hbm-bar" style="width:72%"></div><span class="hbm-val">~25조원</span></div>
          <div class="hbm-row"><span>2026E</span><div class="hbm-bar" style="width:100%"></div><span class="hbm-val">~40조원</span></div>
        </div>
        <div class="ind-note">2022→2026 약 20배 성장 전망. AI 서버 1대당 HBM 탑재량도 지속 증가</div>
      </div>

      <div class="ind-block">
        <div class="ind-title">🥇 HBM 시장 점유율 (2024년 추정)</div>
        <div class="share-list">
          <div class="share-item">
            <span class="share-name" style="color:#FF6B00">SK하이닉스</span>
            <div class="share-bar-wrap"><div class="share-bar" style="width:50%;background:#FF6B00"></div></div>
            <span class="share-pct">~50%</span>
          </div>
          <div class="share-item">
            <span class="share-name" style="color:#1976D2">삼성전자</span>
            <div class="share-bar-wrap"><div class="share-bar" style="width:40%;background:#1976D2"></div></div>
            <span class="share-pct">~40%</span>
          </div>
          <div class="share-item">
            <span class="share-name" style="color:#6b7280">Micron</span>
            <div class="share-bar-wrap"><div class="share-bar" style="width:10%;background:#6b7280"></div></div>
            <span class="share-pct">~10%</span>
          </div>
        </div>
        <div class="ind-note">삼성이 2025년 HBM4에서 점유율 회복 도전 중. Micron은 빠르게 추격</div>
      </div>

      <div class="ind-block">
        <div class="ind-title">🔄 메모리 사이클 현황</div>
        <div class="cycle-wrap">
          <div class="cycle-track">
            <div class="cycle-labels">
              <span>침체</span><span>저점</span><span>회복</span><span>호황</span><span>과열</span>
            </div>
            <div class="cycle-bar">
              <div class="cycle-marker" style="left:62%">▼<br><span>현재</span></div>
            </div>
          </div>
        </div>
        <div class="ind-note">2023년 저점 통과 후 회복 국면. HBM 수요로 DRAM 가격 강세. NAND는 더딘 회복</div>
      </div>

      <div class="ind-block">
        <div class="ind-title">🤖 글로벌 AI 인프라 투자 (Capex)</div>
        <div class="capex-list">
          <div class="capex-item"><span class="capex-co">Microsoft</span><div class="capex-bar-w"><div class="capex-bar" style="width:90%"></div></div><span class="capex-amt">~$80B</span></div>
          <div class="capex-item"><span class="capex-co">Google</span><div class="capex-bar-w"><div class="capex-bar" style="width:72%"></div></div><span class="capex-amt">~$75B</span></div>
          <div class="capex-item"><span class="capex-co">Amazon</span><div class="capex-bar-w"><div class="capex-bar" style="width:75%"></div></div><span class="capex-amt">~$75B</span></div>
          <div class="capex-item"><span class="capex-co">Meta</span><div class="capex-bar-w"><div class="capex-bar" style="width:65%"></div></div><span class="capex-amt">~$60B</span></div>
        </div>
        <div class="ind-note">2025년 글로벌 빅테크 AI Capex 합계 $290B+ 전망. 반도체 수요의 강력한 버팀목</div>
      </div>

      <div class="ind-block">
        <div class="ind-title">⚖️ 핵심 모니터링 지표</div>
        <div class="monitor-list">
          <div class="monitor-item">📌 <strong>DRAM 현물가격</strong> — 주간 변동 추적. 고정가 선행지표</div>
          <div class="monitor-item">📌 <strong>HBM 수주 공시</strong> — SK하이닉스·삼성 분기 실적 발표</div>
          <div class="monitor-item">📌 <strong>필라델피아 반도체 지수(SOX)</strong> — 글로벌 센티먼트</div>
          <div class="monitor-item">📌 <strong>NVIDIA 실적 가이던스</strong> — HBM 수요 바로미터</div>
          <div class="monitor-item">📌 <strong>미국 수출규제 업데이트</strong> — BIS Entity List 변동</div>
          <div class="monitor-item">📌 <strong>USD/KRW 환율</strong> — 수출 비중 80%+, 환율 민감도 높음</div>
        </div>
      </div>

      <div class="ind-block">
        <div class="ind-title">🆚 두 종목 투자 성격 비교</div>
        <div class="compare-nature">
          <div class="nat-row"><span class="nat-label">성격</span><span class="nat-sam">안정적 성장주<br>(배당+가치)</span><span class="nat-hyx">공격적 성장주<br>(모멘텀)</span></div>
          <div class="nat-row"><span class="nat-label">변동성</span><span class="nat-sam">낮음~중간</span><span class="nat-hyx">중간~높음</span></div>
          <div class="nat-row"><span class="nat-label">사이클 민감도</span><span class="nat-sam">중간 (복합사업)</span><span class="nat-hyx">매우 높음 (순수 메모리)</span></div>
          <div class="nat-row"><span class="nat-label">추천 보유기간</span><span class="nat-sam">장기 (2년+)</span><span class="nat-hyx">중기 (6~18개월)</span></div>
          <div class="nat-row"><span class="nat-label">배당</span><span class="nat-sam">연 2~3% 안정적</span><span class="nat-hyx">소액 (사업 재투자 우선)</span></div>
        </div>
      </div>

    </div>
  </div>"""

# ─── 메인 HTML 생성 ──────────────────────────────────────────────────────────────

def generate_html(data):
    now = datetime.now(KST)
    sam = data['samsung']; hyx = data['hynix']

    sam_ret = pd.Series(sam['chart']['prices'][:sam['chart']['hist_len']]).pct_change().dropna()
    hyx_ret = pd.Series(hyx['chart']['prices'][:hyx['chart']['hist_len']]).pct_change().dropna()
    min_len = min(len(sam_ret), len(hyx_ret))
    correlation = float(sam_ret.tail(min_len).corr(hyx_ret.tail(min_len)))

    data_json = json.dumps({'samsung': sam, 'hynix': hyx}, ensure_ascii=False)

    sig_colors = {
        'strong-buy': '#00C853', 'buy': '#4CAF50',
        'neutral': '#757575', 'sell': '#FF5722', 'strong-sell': '#B71C1C',
    }
    def badge(s): return sig_colors.get(s['type'], '#757575'), s['overall']
    sam_sc, sam_sl = badge(sam['signal'])
    hyx_sc, hyx_sl = badge(hyx['signal'])

    def fp(v):   return f"{v:,.0f}원" if v is not None else 'N/A'
    def fpct(v):
        if v is None: return 'N/A'
        return f"{'+'if v>0 else ''}{v:.2f}%"
    def fcap(v): return f"{v/1e12:.1f}조원" if v is not None else 'N/A'
    def rec_label(r):
        m = {'buy':'매수','strong_buy':'강력매수','hold':'보유','sell':'매도','underperform':'하회','outperform':'상회'}
        return m.get(str(r or '').lower(), str(r) if r else 'N/A')

    def prange(s):
        lo, hi, pr = s['low_52w'], s['high_52w'], s['price']
        return round((pr-lo)/(hi-lo)*100, 1) if hi != lo else 50.0

    def sig_items_html(sig):
        tc = {'buy':'#10b981','sell':'#ef4444','neutral':'#9ca3af'}
        return ''.join(
            f'<div class="sig-item"><span class="sig-dot" style="background:{tc.get(s["type"],"#9ca3af")}"></span>{s["label"]}</div>'
            for s in sig['details']
        )

    def analyst_block(s):
        if not s.get('analyst_target'): return ''
        upside = round((s['price'] / s['analyst_target'] - 1) * 100, 1)
        uc = '#10b981' if upside > 0 else '#ef4444'
        ul = f"{'▲'if upside>0 else '▼'} {abs(upside):.1f}% {'상승여력'if upside>0 else '하락위험'}"
        return f"""
        <div class="subsection">
          <div class="sub-title">애널리스트 목표가
            <span style="font-size:.72rem;color:#9ca3af">({s["analyst_count"] or "N/A"}명)</span>
          </div>
          <div style="display:flex;gap:18px;align-items:baseline;margin-top:7px;flex-wrap:wrap">
            <span style="color:#ef4444;font-size:.8rem">저 {fp(s["analyst_low"])}</span>
            <span style="font-size:1.05rem;font-weight:700">{fp(s["analyst_target"])}</span>
            <span style="color:#10b981;font-size:.8rem">고 {fp(s["analyst_high"])}</span>
          </div>
          <div style="margin-top:5px;font-size:.78rem;color:#9ca3af">
            투자의견 <strong style="color:#f9fafb">{rec_label(s["recommendation"])}</strong>
            &nbsp;<span style="color:{uc}">{ul}</span>
          </div>
        </div>"""

    def stock_card(s, sc, sl):
        pr = prange(s)
        up = s['change'] > 0; dn = s['change'] < 0
        cc = 'up' if up else ('dn' if dn else '')
        ca = '▲' if up else ('▼' if dn else '―')
        yc = '#ef4444' if s['ytd_return'] > 0 else '#3b82f6'
        return f"""
      <div class="card">
        <div class="stock-header">
          <div class="stock-name">
            <span class="stock-dot" style="background:{s["color"]}"></span>
            <div>
              <div class="stock-title">{s["name"]}</div>
              <div class="stock-ticker">{s["ticker"]}</div>
            </div>
          </div>
          <span class="badge" style="background:{sc}">{sl}</span>
        </div>
        <div class="price-block">
          <div class="price-main">{fp(s["price"])}</div>
          <div class="price-change {cc}">{ca} {abs(s["change"]):,.0f}원 ({fpct(s["change_pct"])})</div>
        </div>
        <div class="week52">
          <div class="sub-title">52주 범위</div>
          <div class="range-track">
            <div class="range-fill" style="width:{pr}%"></div>
            <div class="range-pin" style="left:{pr}%"></div>
          </div>
          <div class="range-labels"><span>저 {fp(s["low_52w"])}</span><span>고 {fp(s["high_52w"])}</span></div>
        </div>
        <div class="metrics">
          <div class="metric"><div class="m-label">1년 수익률</div><div class="m-val" style="color:{yc}">{fpct(s["ytd_return"])}</div></div>
          <div class="metric"><div class="m-label">30일 변동성</div><div class="m-val">{s["volatility"]:.1f}%</div></div>
          <div class="metric"><div class="m-label">PER</div><div class="m-val">{f'{s["pe"]:.1f}x' if s["pe"] else 'N/A'}</div></div>
          <div class="metric"><div class="m-label">PBR</div><div class="m-val">{f'{s["pb"]:.2f}x' if s["pb"] else 'N/A'}</div></div>
          <div class="metric"><div class="m-label">시가총액</div><div class="m-val">{fcap(s["market_cap"])}</div></div>
          <div class="metric"><div class="m-label">배당수익률</div><div class="m-val">{f'{s["dividend_yield"]*100:.2f}%' if s["dividend_yield"] else 'N/A'}</div></div>
        </div>
        {analyst_block(s)}
        <div class="subsection">
          <div class="sub-title">기술 분석 신호</div>
          <div class="sig-list" style="margin-top:8px">{sig_items_html(s["signal"])}</div>
        </div>
      </div>"""

    corr_desc = ('높음 — 함께 움직이는 경향 강함' if abs(correlation) > 0.7
                 else '중간 — 어느 정도 분산 효과' if abs(correlation) > 0.4
                 else '낮음 — 독립적 움직임')

    def comp_row(label, sv, hv, sc='', hc=''):
        return f"""<div class="comp-row">
          <div class="cv-left" style="color:{sc if sc else 'inherit'}">{sv}</div>
          <div class="cv-mid">{label}</div>
          <div class="cv-right" style="color:{hc if hc else 'inherit'}">{hv}</div>
        </div>"""

    comp_rows = (
        comp_row('당일 등락', fpct(sam['change_pct']), fpct(hyx['change_pct']),
                 '#ef4444' if sam['change_pct']>0 else '#3b82f6',
                 '#ef4444' if hyx['change_pct']>0 else '#3b82f6') +
        comp_row('1년 수익률', fpct(sam['ytd_return']), fpct(hyx['ytd_return']),
                 '#ef4444' if sam['ytd_return']>0 else '#3b82f6',
                 '#ef4444' if hyx['ytd_return']>0 else '#3b82f6') +
        comp_row('30일 변동성', f"{sam['volatility']:.1f}%", f"{hyx['volatility']:.1f}%") +
        comp_row('PER', f"{sam['pe']:.1f}x" if sam['pe'] else 'N/A', f"{hyx['pe']:.1f}x" if hyx['pe'] else 'N/A') +
        comp_row('PBR', f"{sam['pb']:.2f}x" if sam['pb'] else 'N/A', f"{hyx['pb']:.2f}x" if hyx['pb'] else 'N/A') +
        comp_row('시가총액', fcap(sam['market_cap']), fcap(hyx['market_cap'])) +
        comp_row('목표가(평균)', fp(sam.get('analyst_target')), fp(hyx.get('analyst_target'))) +
        comp_row('투자 신호', sam['signal']['overall'], hyx['signal']['overall'])
    )

    deep_html    = deep_analysis_section(sam, hyx)
    scen_html    = scenarios_section(sam, hyx)
    indust_html  = industry_section()

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>삼성전자 &amp; SK하이닉스 주식 대시보드</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#0a0e1a;--surf:#111827;--surf2:#1f2937;--brd:#374151;--txt:#f9fafb;--dim:#9ca3af;--grn:#10b981;--red:#ef4444;--blu:#3b82f6}}
body{{font-family:-apple-system,'Segoe UI','Noto Sans KR',sans-serif;background:var(--bg);color:var(--txt);min-height:100vh}}
.header{{background:linear-gradient(135deg,#0f172a,#1e293b);padding:18px 24px;border-bottom:1px solid var(--brd);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px}}
.header h1{{font-size:1.35rem;font-weight:800}}
.header .sub{{color:var(--dim);font-size:.8rem;margin-top:3px}}
.upd{{background:var(--surf2);border:1px solid var(--brd);border-radius:8px;padding:7px 14px;font-size:.78rem;color:var(--dim)}}
.wrap{{max-width:1400px;margin:0 auto;padding:20px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:20px}}
@media(max-width:860px){{.grid2{{grid-template-columns:1fr}}}}
.card{{background:var(--surf);border:1px solid var(--brd);border-radius:14px;padding:22px}}
.section-card{{background:var(--surf);border:1px solid var(--brd);border-radius:14px;padding:22px;margin-bottom:20px}}
.section-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;flex-wrap:wrap;gap:10px}}
.section-title{{font-size:1rem;font-weight:700}}
.stock-header{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:18px}}
.stock-name{{display:flex;align-items:center;gap:10px}}
.stock-dot{{width:11px;height:11px;border-radius:50%;flex-shrink:0;margin-top:3px}}
.stock-title{{font-size:1.15rem;font-weight:700}}
.stock-ticker{{color:var(--dim);font-size:.8rem;margin-top:2px}}
.badge{{padding:5px 13px;border-radius:16px;font-size:.78rem;font-weight:600;color:#fff}}
.price-block{{margin-bottom:16px}}
.price-main{{font-size:1.9rem;font-weight:800;letter-spacing:-.5px}}
.price-change{{margin-top:4px;font-size:.9rem}}
.price-change.up{{color:var(--red)}}.price-change.dn{{color:var(--blu)}}
.week52{{margin-bottom:16px}}
.range-track{{position:relative;background:var(--surf2);height:6px;border-radius:3px;margin:8px 0}}
.range-fill{{position:absolute;height:100%;background:linear-gradient(90deg,var(--blu),var(--grn));border-radius:3px}}
.range-pin{{position:absolute;top:-5px;width:16px;height:16px;border-radius:50%;background:#fff;border:2px solid var(--blu);transform:translateX(-50%)}}
.range-labels{{display:flex;justify-content:space-between;font-size:.73rem;color:var(--dim)}}
.metrics{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px}}
.metric{{background:var(--surf2);border-radius:9px;padding:10px 12px}}
.m-label{{font-size:.7rem;color:var(--dim);text-transform:uppercase;letter-spacing:.4px}}
.m-val{{font-size:.95rem;font-weight:600;margin-top:3px}}
.subsection{{margin-top:16px;padding-top:16px;border-top:1px solid var(--brd)}}
.sub-title{{font-size:.72rem;text-transform:uppercase;letter-spacing:.8px;color:var(--dim);font-weight:600;margin-bottom:8px}}
.sig-list{{display:flex;flex-direction:column;gap:5px}}
.sig-item{{display:flex;align-items:center;gap:7px;font-size:.78rem}}
.sig-dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0}}
.tabs{{display:flex;gap:6px}}
.tab{{padding:6px 16px;border-radius:7px;border:1px solid var(--brd);background:transparent;color:var(--dim);cursor:pointer;font-size:.82rem;transition:all .15s}}
.tab.on{{background:var(--surf2);color:var(--txt);border-color:var(--blu)}}
/* 강점·리스크 */
.str-risk-grid{{display:grid;grid-template-columns:1fr 1fr;gap:28px;margin-top:4px}}
@media(max-width:720px){{.str-risk-grid{{grid-template-columns:1fr}}}}
.pane-title{{font-size:.82rem;font-weight:700;color:var(--dim);text-transform:uppercase;letter-spacing:.6px;margin-bottom:14px}}
.pane-sub{{font-size:.7rem;font-weight:400;margin-left:4px}}
.sfactor{{margin-bottom:14px}}
.sf-row{{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px}}
.sf-name{{font-size:.83rem;font-weight:500}}
.sf-score{{font-size:.85rem;font-weight:700}}
.sf-track{{background:var(--surf2);height:6px;border-radius:3px;overflow:hidden}}
.sf-fill{{height:100%;border-radius:3px;transition:width .4s}}
.sf-desc{{font-size:.72rem;color:var(--dim);margin-top:4px;line-height:1.45}}
.rfactor{{margin-bottom:14px;background:var(--surf2);border-radius:9px;padding:10px 12px}}
.rf-row{{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px}}
.rf-name{{font-size:.83rem;font-weight:600}}
.rf-badge{{font-size:.7rem;font-weight:600;padding:2px 9px;border-radius:12px}}
.rf-desc{{font-size:.72rem;color:var(--dim);line-height:1.45}}
/* 시나리오 */
.scen-note{{font-size:.75rem;color:var(--dim);margin-bottom:16px}}
.scen-current{{font-size:.85rem;color:var(--dim);margin-bottom:14px}}
.scen-current strong{{color:var(--txt)}}
.scen-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}}
@media(max-width:680px){{.scen-grid{{grid-template-columns:1fr}}}}
.scen-col{{background:var(--surf2);border-radius:12px;padding:18px;display:flex;flex-direction:column;gap:6px}}
.scen-label{{font-size:.82rem;font-weight:700}}
.scen-prob{{font-size:.72rem;font-weight:600}}
.scen-price{{font-size:1.4rem;font-weight:800;margin-top:4px}}
.scen-pct{{font-size:1rem;font-weight:700}}
.scen-list{{margin-top:10px;padding-left:14px;font-size:.75rem;color:var(--dim);line-height:1.7}}
/* 산업 분석 */
.industry-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
@media(max-width:800px){{.industry-grid{{grid-template-columns:1fr}}}}
.ind-block{{background:var(--surf2);border-radius:11px;padding:16px}}
.ind-title{{font-size:.82rem;font-weight:700;color:var(--dim);text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px}}
.ind-note{{font-size:.72rem;color:var(--dim);margin-top:10px;line-height:1.5}}
.hbm-bar-wrap{{display:flex;flex-direction:column;gap:7px}}
.hbm-row{{display:flex;align-items:center;gap:8px;font-size:.78rem}}
.hbm-row span:first-child{{width:38px;color:var(--dim);flex-shrink:0}}
.hbm-bar{{height:14px;background:linear-gradient(90deg,#1976D2,#FF6B00);border-radius:3px;transition:width .5s}}
.hbm-val{{font-size:.75rem;color:var(--txt);margin-left:4px;white-space:nowrap}}
.share-list{{display:flex;flex-direction:column;gap:10px}}
.share-item{{display:flex;align-items:center;gap:8px}}
.share-name{{width:68px;font-size:.78rem;font-weight:600;flex-shrink:0}}
.share-bar-wrap{{flex:1;background:#374151;height:14px;border-radius:3px;overflow:hidden}}
.share-bar{{height:100%;border-radius:3px}}
.share-pct{{width:36px;font-size:.78rem;color:var(--dim);text-align:right}}
.cycle-wrap{{padding:10px 0}}
.cycle-labels{{display:flex;justify-content:space-between;font-size:.68rem;color:var(--dim);margin-bottom:6px}}
.cycle-bar{{position:relative;height:12px;background:linear-gradient(90deg,#374151 0%,#3b82f6 25%,#10b981 50%,#f59e0b 75%,#ef4444 100%);border-radius:6px}}
.cycle-marker{{position:absolute;top:-22px;transform:translateX(-50%);font-size:.7rem;text-align:center;color:#fff;white-space:nowrap}}
.cycle-marker span{{font-size:.65rem;color:var(--dim)}}
.capex-list{{display:flex;flex-direction:column;gap:8px}}
.capex-item{{display:flex;align-items:center;gap:8px}}
.capex-co{{width:72px;font-size:.78rem;font-weight:600;flex-shrink:0}}
.capex-bar-w{{flex:1;background:#374151;height:12px;border-radius:3px;overflow:hidden}}
.capex-bar{{height:100%;background:linear-gradient(90deg,#8b5cf6,#3b82f6);border-radius:3px}}
.capex-amt{{width:42px;font-size:.75rem;color:var(--dim);text-align:right;white-space:nowrap}}
.monitor-list{{display:flex;flex-direction:column;gap:8px}}
.monitor-item{{font-size:.78rem;color:var(--dim);line-height:1.5}}
.monitor-item strong{{color:var(--txt)}}
.compare-nature{{display:flex;flex-direction:column;gap:0}}
.nat-row{{display:grid;grid-template-columns:70px 1fr 1fr;gap:8px;padding:7px 0;border-bottom:1px solid var(--brd);font-size:.78rem}}
.nat-row:last-child{{border-bottom:none}}
.nat-label{{color:var(--dim);font-weight:600}}
.nat-sam{{color:#64B5F6}}
.nat-hyx{{color:#FFB74D}}
/* 차트 공통 */
.chart-wrap{{position:relative}}
.chart-card{{background:var(--surf);border:1px solid var(--brd);border-radius:14px;padding:22px;margin-bottom:20px}}
.chart-title{{font-size:1rem;font-weight:700}}
/* 비교 */
.comp-card{{background:var(--surf);border:1px solid var(--brd);border-radius:14px;padding:22px;margin-bottom:20px}}
.comp-head{{display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:center;padding:8px 0;border-bottom:2px solid var(--brd);margin-bottom:4px}}
.comp-row{{display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:center;padding:9px 0;border-bottom:1px solid var(--surf2)}}
.comp-row:last-child{{border-bottom:none}}
.cv-left{{text-align:right;font-size:.87rem;font-weight:500}}
.cv-right{{text-align:left;font-size:.87rem;font-weight:500}}
.cv-mid{{text-align:center;font-size:.7rem;color:var(--dim);text-transform:uppercase;white-space:nowrap}}
.corr-note{{margin-top:14px;background:var(--surf2);border-radius:9px;padding:12px 16px;font-size:.82rem;color:var(--dim)}}
/* 인사이트 */
.insight-card{{background:var(--surf);border:1px solid var(--brd);border-radius:14px;padding:22px;margin-bottom:20px}}
.insight-item{{display:flex;gap:14px;padding:12px 0;border-bottom:1px solid var(--surf2)}}
.insight-item:last-child{{border-bottom:none}}
.insight-icon{{font-size:1.2rem;flex-shrink:0;margin-top:1px}}
.i-head{{font-weight:600;margin-bottom:4px;font-size:.88rem}}
.i-desc{{font-size:.8rem;color:var(--dim);line-height:1.55}}
.footer{{text-align:center;padding:20px;color:var(--dim);font-size:.73rem;border-top:1px solid var(--brd)}}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>📈 삼성전자 &amp; SK하이닉스 주식 대시보드</h1>
    <div class="sub">매일 오전 6시(KST) 자동 업데이트 · 1년 예측 + 강점·리스크·시나리오 분석 포함</div>
  </div>
  <div class="upd">🕐 업데이트: {now.strftime('%Y-%m-%d %H:%M')} KST</div>
</div>

<div class="wrap">
  <!-- 종목 카드 -->
  <div class="grid2">
    {stock_card(sam, sam_sc, sam_sl)}
    {stock_card(hyx, hyx_sc, hyx_sl)}
  </div>

  {deep_html}

  <!-- 가격 차트 -->
  <div class="chart-card">
    <div class="section-head">
      <div class="chart-title">📊 주가 차트 (1년 실적 + 1년 예측선)</div>
      <div class="tabs">
        <button class="tab on"  onclick="switchStock('samsung')">삼성전자</button>
        <button class="tab off" onclick="switchStock('hynix')">SK하이닉스</button>
      </div>
    </div>
    <div class="chart-wrap" style="height:380px"><canvas id="priceChart"></canvas></div>
  </div>

  <!-- 기술 지표 -->
  <div class="grid2">
    <div class="chart-card" style="margin-bottom:0">
      <div class="chart-title" style="margin-bottom:14px">RSI (14일)</div>
      <div class="chart-wrap" style="height:150px"><canvas id="rsiChart"></canvas></div>
    </div>
    <div class="chart-card" style="margin-bottom:0">
      <div class="chart-title" style="margin-bottom:14px">MACD (12, 26, 9)</div>
      <div class="chart-wrap" style="height:150px"><canvas id="macdChart"></canvas></div>
    </div>
  </div>
  <div style="margin-bottom:20px"></div>

  <!-- 거래량 -->
  <div class="chart-card">
    <div class="chart-title" style="margin-bottom:14px">📊 거래량</div>
    <div class="chart-wrap" style="height:120px"><canvas id="volChart"></canvas></div>
  </div>

  {scen_html}

  <!-- 비교 -->
  <div class="comp-card">
    <div class="section-title" style="margin-bottom:18px">⚖️ 두 종목 비교</div>
    <div class="comp-head">
      <div class="cv-left" style="font-weight:700;color:{sam['color']}">{sam['name']}</div>
      <div class="cv-mid">항목</div>
      <div class="cv-right" style="font-weight:700;color:{hyx['color']}">{hyx['name']}</div>
    </div>
    {comp_rows}
    <div class="corr-note">📐 상관계수 <strong style="color:var(--txt)">{correlation:.3f}</strong> — {corr_desc}</div>
  </div>

  {indust_html}

  <!-- 인사이트 -->
  <div class="insight-card">
    <div class="section-title" style="margin-bottom:18px">💡 투자 인사이트</div>
    <div class="insight-item">
      <div class="insight-icon">🔮</div>
      <div><div class="i-head">1년 예측 방법론</div>
      <div class="i-desc">최근 6개월 주가 선형 추세와 애널리스트 컨센서스 목표가를 가중 혼합 산출. 예측값은 참고용이며 시장 변동성에 따라 실제 주가와 크게 다를 수 있습니다.</div></div>
    </div>
    <div class="insight-item">
      <div class="insight-icon">📅</div>
      <div><div class="i-head">주요 이벤트 일정</div>
      <div class="i-desc">삼성·하이닉스 분기 실적: 매 분기 말 다음달 초 / CES(1월) · MWC(2-3월) · Computex(5월) · Hot Chips(8월) / FOMC 금리 결정·CPI 발표 · MSCI 비중 조정</div></div>
    </div>
    <div class="insight-item">
      <div class="insight-icon">⚠️</div>
      <div><div class="i-head">면책 조항</div>
      <div class="i-desc">본 대시보드는 공개 데이터 기반 참고 자료이며, 투자 권유가 아닙니다. 모든 투자 판단과 책임은 투자자 본인에게 있습니다. 과거 수익률이 미래를 보장하지 않습니다.</div></div>
    </div>
  </div>
</div>

<div class="footer">
  데이터 출처: Yahoo Finance (yfinance) · GitHub Actions 자동 생성 · 매일 오전 6시(KST) 업데이트
</div>

<script>
const DATA = {data_json};
let cur = 'samsung';
let PC, RC, MC, VC;

const BO = {{
  responsive:true, maintainAspectRatio:false, animation:false,
  plugins:{{
    legend:{{labels:{{color:'#9ca3af',font:{{size:10}}}}}},
    tooltip:{{mode:'index',intersect:false,backgroundColor:'#1f2937',titleColor:'#f9fafb',bodyColor:'#9ca3af',borderColor:'#374151',borderWidth:1}}
  }},
  scales:{{
    x:{{ticks:{{color:'#6b7280',maxTicksLimit:9,font:{{size:9}}}},grid:{{color:'#1a2235'}}}},
    y:{{ticks:{{color:'#6b7280',font:{{size:9}}}},grid:{{color:'#1a2235'}}}}
  }}
}};

function initCharts(key) {{
  [PC,RC,MC,VC].forEach(c=>c&&c.destroy());
  const d=DATA[key], c=d.chart;
  PC=new Chart(document.getElementById('priceChart'),{{
    type:'line',
    data:{{labels:c.dates,datasets:[
      {{label:'종가',data:c.prices,borderColor:d.color,borderWidth:2,pointRadius:0,fill:false,tension:.1,spanGaps:false}},
      {{label:'1년 예측',data:c.pred,borderColor:d.color_pred,borderWidth:2,borderDash:[7,4],pointRadius:0,fill:false,tension:.2,spanGaps:false}},
      {{label:'20일선',data:c.sma20,borderColor:'#f59e0b',borderWidth:1,pointRadius:0,fill:false,tension:.1,spanGaps:false}},
      {{label:'50일선',data:c.sma50,borderColor:'#8b5cf6',borderWidth:1,pointRadius:0,fill:false,tension:.1,spanGaps:false}},
      {{label:'200일선',data:c.sma200,borderColor:'#ef4444',borderWidth:1,pointRadius:0,fill:false,tension:.1,spanGaps:false}},
      {{label:'볼린저↑',data:c.bb_upper,borderColor:'#4b5563',borderWidth:1,borderDash:[3,3],pointRadius:0,fill:false,spanGaps:false}},
      {{label:'볼린저↓',data:c.bb_lower,borderColor:'#4b5563',borderWidth:1,borderDash:[3,3],pointRadius:0,fill:false,spanGaps:false}},
    ]}},
    options:{{...BO,plugins:{{...BO.plugins,tooltip:{{...BO.plugins.tooltip,callbacks:{{label:ctx=>ctx.raw!=null?' '+ctx.dataset.label+': '+ctx.raw.toLocaleString('ko-KR')+'원':null}}}}}},scales:{{x:BO.scales.x,y:{{...BO.scales.y,ticks:{{color:'#6b7280',font:{{size:9}},callback:v=>v!=null?v.toLocaleString('ko-KR')+'원':''}}}}}}}}
  }});
  RC=new Chart(document.getElementById('rsiChart'),{{
    type:'line',
    data:{{labels:c.dates,datasets:[
      {{label:'RSI',data:c.rsi,borderColor:'#a78bfa',borderWidth:1.5,pointRadius:0,fill:false,spanGaps:false}},
      {{label:'과매수(70)',data:c.dates.map(()=>70),borderColor:'#ef4444',borderWidth:1,borderDash:[4,4],pointRadius:0,fill:false}},
      {{label:'과매도(30)',data:c.dates.map(()=>30),borderColor:'#3b82f6',borderWidth:1,borderDash:[4,4],pointRadius:0,fill:false}},
    ]}},
    options:{{...BO,scales:{{x:BO.scales.x,y:{{...BO.scales.y,min:0,max:100}}}}}}
  }});
  MC=new Chart(document.getElementById('macdChart'),{{
    data:{{labels:c.dates,datasets:[
      {{type:'bar',label:'히스토그램',data:c.macd_hist,backgroundColor:c.macd_hist.map(v=>v>=0?'#10b98166':'#ef444466')}},
      {{type:'line',label:'MACD',data:c.macd,borderColor:'#3b82f6',borderWidth:1.5,pointRadius:0,fill:false,spanGaps:false}},
      {{type:'line',label:'시그널',data:c.macd_signal,borderColor:'#f97316',borderWidth:1.5,pointRadius:0,fill:false,spanGaps:false}},
    ]}},options:BO
  }});
  VC=new Chart(document.getElementById('volChart'),{{
    type:'bar',
    data:{{labels:c.dates,datasets:[{{label:'거래량',data:c.volume,backgroundColor:d.color+'66',borderWidth:0}}]}},
    options:{{...BO,scales:{{x:BO.scales.x,y:{{...BO.scales.y,ticks:{{color:'#6b7280',font:{{size:9}},callback:v=>v>=1e6?(v/1e6).toFixed(0)+'M':v.toLocaleString()}}}}}}}}
  }});
}}

function switchStock(key) {{
  cur=key;
  document.querySelectorAll('.tab').forEach((b,i)=>b.classList.toggle('on',(key==='samsung'&&i===0)||(key==='hynix'&&i===1)));
  initCharts(key);
}}

function switchAnalysis(key, btn) {{
  document.getElementById('anal-samsung').style.display = key==='samsung'?'':'none';
  document.getElementById('anal-hynix').style.display   = key==='hynix'  ?'':'none';
  btn.closest('.tabs').querySelectorAll('.tab').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
}}

function switchScen(key, btn) {{
  document.getElementById('scen-samsung').style.display = key==='samsung'?'':'none';
  document.getElementById('scen-hynix').style.display   = key==='hynix'  ?'':'none';
  btn.closest('.tabs').querySelectorAll('.tab').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
}}

initCharts('samsung');
</script>
</body>
</html>"""


def main():
    print(f"[{datetime.now(KST).strftime('%Y-%m-%d %H:%M')} KST] 대시보드 생성 시작")
    data = {}
    for key in STOCKS:
        result = process_stock(key)
        if result:
            data[key] = result
        else:
            print(f"  {STOCKS[key]['name']} 로드 실패", file=sys.stderr)

    if len(data) < 2:
        print("데이터 부족 — 중단", file=sys.stderr)
        sys.exit(1)

    html = generate_html(data)
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html)

    print("index.html 생성 완료")
    for k, d in data.items():
        s = '+' if d['change'] > 0 else ''
        print(f"  {d['name']}: {d['price']:,.0f}원 ({s}{d['change_pct']:.2f}%) | {d['signal']['overall']}")


if __name__ == '__main__':
    main()

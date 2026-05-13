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
    'samsung': {
        'ticker': '005930.KS',
        'name': '삼성전자',
        'color': '#1976D2',
        'color_pred': '#64B5F6',
    },
    'hynix': {
        'ticker': '000660.KS',
        'name': 'SK하이닉스',
        'color': '#FF6B00',
        'color_pred': '#FFB74D',
    },
}


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
    analyst_high = info.get('targetHighPrice')
    analyst_low = info.get('targetLowPrice')

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
    close_1y = hist_1y['Close']
    close_full = hist_full['Close']
    price = close_1y.iloc[-1]

    rsi_val = calc_rsi(close_1y).iloc[-1]
    macd_line, macd_sig, _ = calc_macd(close_1y)
    sma20 = close_1y.rolling(20).mean().iloc[-1]
    sma50 = close_1y.rolling(50).mean().iloc[-1]
    sma200 = close_full.rolling(200).mean().iloc[-1]
    bb_upper, _, bb_lower = calc_bb(close_1y)

    score = 0
    details = []

    if not np.isnan(rsi_val):
        if rsi_val < 30:
            score += 2
            details.append({'label': f'RSI 과매도 ({rsi_val:.1f})', 'type': 'buy'})
        elif rsi_val > 70:
            score -= 2
            details.append({'label': f'RSI 과매수 ({rsi_val:.1f})', 'type': 'sell'})
        else:
            details.append({'label': f'RSI 중립 ({rsi_val:.1f})', 'type': 'neutral'})

    if macd_line.iloc[-1] > macd_sig.iloc[-1]:
        score += 1
        details.append({'label': 'MACD 골든크로스', 'type': 'buy'})
    else:
        score -= 1
        details.append({'label': 'MACD 데드크로스', 'type': 'sell'})

    if price > sma20:
        score += 1
        details.append({'label': f'20일선 위 ({sma20:,.0f}원)', 'type': 'buy'})
    else:
        score -= 1
        details.append({'label': f'20일선 아래 ({sma20:,.0f}원)', 'type': 'sell'})

    if price > sma50:
        score += 1
        details.append({'label': f'50일선 위 ({sma50:,.0f}원)', 'type': 'buy'})
    else:
        score -= 1
        details.append({'label': f'50일선 아래 ({sma50:,.0f}원)', 'type': 'sell'})

    if not np.isnan(sma200):
        if price > sma200:
            score += 1
            details.append({'label': f'200일선 위 ({sma200:,.0f}원)', 'type': 'buy'})
        else:
            score -= 1
            details.append({'label': f'200일선 아래 ({sma200:,.0f}원)', 'type': 'sell'})

    if not np.isnan(bb_lower.iloc[-1]) and price <= bb_lower.iloc[-1]:
        score += 1
        details.append({'label': '볼린저 하단 지지', 'type': 'buy'})
    elif not np.isnan(bb_upper.iloc[-1]) and price >= bb_upper.iloc[-1]:
        score -= 1
        details.append({'label': '볼린저 상단 저항', 'type': 'sell'})

    if score >= 4:
        overall, stype = '강력 매수', 'strong-buy'
    elif score >= 2:
        overall, stype = '매수', 'buy'
    elif score <= -4:
        overall, stype = '강력 매도', 'strong-sell'
    elif score <= -2:
        overall, stype = '매도', 'sell'
    else:
        overall, stype = '중립', 'neutral'

    def sf(v):
        return round(float(v), 1) if not np.isnan(v) else None

    return {
        'overall': overall, 'type': stype, 'score': score, 'details': details,
        'rsi': sf(rsi_val),
        'sma20': round(float(sma20), 0), 'sma50': round(float(sma50), 0),
        'sma200': round(float(sma200), 0) if not np.isnan(sma200) else None,
        'macd': round(float(macd_line.iloc[-1]), 2),
        'macd_signal': round(float(macd_sig.iloc[-1]), 2),
        'bb_upper': round(float(bb_upper.iloc[-1]), 0) if not np.isnan(bb_upper.iloc[-1]) else None,
        'bb_lower': round(float(bb_lower.iloc[-1]), 0) if not np.isnan(bb_lower.iloc[-1]) else None,
    }


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
    prev = float(close.iloc[-2]) if len(close) > 1 else price

    sma20_s = close.rolling(20).mean()
    sma50_s = close.rolling(50).mean()
    sma200_full = hist_full['Close'].rolling(200).mean()
    sma200_1y = sma200_full[sma200_full.index >= one_year_ago]
    bb_upper, _, bb_lower = calc_bb(close)
    rsi_s = calc_rsi(close)
    macd_l, macd_sig_s, macd_h = calc_macd(close)

    future_dates, predicted, a_target, a_high, a_low = predict_prices(hist_full, info)
    signal = get_signal(hist_full, hist_1y)

    returns = close.pct_change().dropna()
    volatility = float(returns.tail(30).std() * np.sqrt(252) * 100)
    ytd_return = float((price - float(close.iloc[0])) / float(close.iloc[0]) * 100)

    def sf(v):
        if v is None:
            return None
        try:
            f = float(v)
            return None if np.isnan(f) else f
        except Exception:
            return None

    hist_len = len(hist_1y)
    pred_len = len(future_dates)
    # Price chart: combine historical + prediction with null overlap
    all_dates = [d.strftime('%Y-%m-%d') for d in hist_1y.index] + \
                [d.strftime('%Y-%m-%d') for d in future_dates]
    prices_all = clean_prices(close.tolist()) + [None] * pred_len
    pred_all = [None] * (hist_len - 1) + \
               [round(price, 0)] + \
               [round(float(v), 0) for v in predicted]
    sma20_all = clean_prices(sma20_s.tolist()) + [None] * pred_len
    sma50_all = clean_prices(sma50_s.tolist()) + [None] * pred_len
    sma200_all = clean_prices(sma200_1y.tolist()) + [None] * pred_len
    bb_upper_all = clean_prices(bb_upper.tolist()) + [None] * pred_len
    bb_lower_all = clean_prices(bb_lower.tolist()) + [None] * pred_len
    volume_all = [int(v) for v in hist_1y['Volume'].tolist()] + [None] * pred_len
    rsi_all = clean_series(rsi_s.tolist()) + [None] * pred_len
    macd_all = clean_series(macd_l.tolist()) + [None] * pred_len
    macd_sig_all = clean_series(macd_sig_s.tolist()) + [None] * pred_len
    macd_h_all = clean_series(macd_h.tolist()) + [None] * pred_len

    return {
        'key': key,
        'name': cfg['name'],
        'ticker': cfg['ticker'],
        'color': cfg['color'],
        'color_pred': cfg['color_pred'],
        'price': round(price, 0),
        'change': round(price - prev, 0),
        'change_pct': round((price - prev) / prev * 100, 2),
        'high_52w': round(float(close.max()), 0),
        'low_52w': round(float(close.min()), 0),
        'volume': int(hist_1y['Volume'].iloc[-1]),
        'avg_volume': int(hist_1y['Volume'].mean()),
        'volatility': round(volatility, 1),
        'ytd_return': round(ytd_return, 1),
        'pe': sf(info.get('trailingPE')),
        'pb': sf(info.get('priceToBook')),
        'eps': sf(info.get('trailingEps')),
        'market_cap': sf(info.get('marketCap')),
        'dividend_yield': sf(info.get('dividendYield')),
        'analyst_target': sf(a_target),
        'analyst_high': sf(a_high),
        'analyst_low': sf(a_low),
        'analyst_count': info.get('numberOfAnalystOpinions'),
        'recommendation': info.get('recommendationKey'),
        'signal': signal,
        'chart': {
            'dates': all_dates,
            'hist_len': hist_len,
            'prices': prices_all,
            'pred': pred_all,
            'sma20': sma20_all,
            'sma50': sma50_all,
            'sma200': sma200_all,
            'bb_upper': bb_upper_all,
            'bb_lower': bb_lower_all,
            'volume': volume_all,
            'rsi': rsi_all,
            'macd': macd_all,
            'macd_signal': macd_sig_all,
            'macd_hist': macd_h_all,
        },
    }


def generate_html(data):
    now = datetime.now(KST)
    sam = data['samsung']
    hyx = data['hynix']

    sam_ret = pd.Series(sam['chart']['prices'][:sam['chart']['hist_len']]).pct_change().dropna()
    hyx_ret = pd.Series(hyx['chart']['prices'][:hyx['chart']['hist_len']]).pct_change().dropna()
    min_len = min(len(sam_ret), len(hyx_ret))
    correlation = float(sam_ret.tail(min_len).corr(hyx_ret.tail(min_len)))

    data_json = json.dumps({'samsung': sam, 'hynix': hyx}, ensure_ascii=False)

    sig_colors = {
        'strong-buy': '#00C853', 'buy': '#4CAF50',
        'neutral': '#757575', 'sell': '#FF5722', 'strong-sell': '#B71C1C',
    }

    def badge(s):
        return sig_colors.get(s['type'], '#757575'), s['overall']

    sam_sc, sam_sl = badge(sam['signal'])
    hyx_sc, hyx_sl = badge(hyx['signal'])

    def fp(v):
        return f"{v:,.0f}원" if v is not None else 'N/A'

    def fpct(v):
        if v is None:
            return 'N/A'
        sign = '+' if v > 0 else ''
        return f"{sign}{v:.2f}%"

    def fcap(v):
        if v is None:
            return 'N/A'
        return f"{v/1e12:.1f}조원"

    def rec_label(r):
        m = {'buy': '매수', 'strong_buy': '강력매수', 'hold': '보유',
             'sell': '매도', 'underperform': '하회', 'outperform': '상회'}
        return m.get(str(r or '').lower(), str(r) if r else 'N/A')

    def prange(stock):
        lo, hi, pr = stock['low_52w'], stock['high_52w'], stock['price']
        if hi == lo:
            return 50.0
        return round((pr - lo) / (hi - lo) * 100, 1)

    def signal_items_html(sig):
        type_color = {'buy': '#10b981', 'sell': '#ef4444', 'neutral': '#9ca3af'}
        items = ''
        for s in sig['details']:
            c = type_color.get(s['type'], '#9ca3af')
            items += f'<div class="sig-item"><span class="sig-dot" style="background:{c}"></span>{s["label"]}</div>'
        return items

    def analyst_section(stock):
        if not stock.get('analyst_target'):
            return ''
        upside = round((stock['price'] / stock['analyst_target'] - 1) * 100, 1)
        upside_label = f"{'▲' if upside > 0 else '▼'} {abs(upside):.1f}% {'상승여력' if upside > 0 else '하락위험'}"
        upside_color = '#10b981' if upside > 0 else '#ef4444'
        return f'''
        <div class="subsection">
          <div class="sub-title">애널리스트 목표가
            <span style="font-size:0.75rem;color:#9ca3af;font-weight:400">({stock["analyst_count"] or "N/A"}명)</span>
          </div>
          <div style="display:flex;gap:20px;align-items:baseline;margin-top:8px;flex-wrap:wrap">
            <span style="color:#ef4444;font-size:0.82rem">저 {fp(stock["analyst_low"])}</span>
            <span style="font-size:1.05rem;font-weight:700">{fp(stock["analyst_target"])}</span>
            <span style="color:#10b981;font-size:0.82rem">고 {fp(stock["analyst_high"])}</span>
          </div>
          <div style="margin-top:6px;font-size:0.8rem;color:#9ca3af">
            투자의견: <strong style="color:#f9fafb">{rec_label(stock["recommendation"])}</strong>
            &nbsp;&nbsp;<span style="color:{upside_color}">{upside_label}</span>
          </div>
        </div>'''

    def stock_card(s, sig_color, sig_label):
        pr = prange(s)
        up = s['change'] > 0
        dn = s['change'] < 0
        change_cls = 'up' if up else ('dn' if dn else '')
        change_arrow = '▲' if up else ('▼' if dn else '―')
        ytd_color = '#ef4444' if s['ytd_return'] > 0 else '#3b82f6'

        return f'''
      <div class="card">
        <div class="stock-header">
          <div class="stock-name">
            <span class="stock-dot" style="background:{s["color"]}"></span>
            <div>
              <div class="stock-title">{s["name"]}</div>
              <div class="stock-ticker">{s["ticker"]}</div>
            </div>
          </div>
          <span class="badge" style="background:{sig_color}">{sig_label}</span>
        </div>

        <div class="price-block">
          <div class="price-main">{fp(s["price"])}</div>
          <div class="price-change {change_cls}">
            {change_arrow} {abs(s["change"]):,.0f}원 ({fpct(s["change_pct"])})
          </div>
        </div>

        <div class="week52">
          <div class="sub-title">52주 범위</div>
          <div class="range-track">
            <div class="range-fill" style="width:{pr}%"></div>
            <div class="range-pin" style="left:{pr}%"></div>
          </div>
          <div class="range-labels">
            <span>저 {fp(s["low_52w"])}</span><span>고 {fp(s["high_52w"])}</span>
          </div>
        </div>

        <div class="metrics">
          <div class="metric">
            <div class="m-label">1년 수익률</div>
            <div class="m-val" style="color:{ytd_color}">{fpct(s["ytd_return"])}</div>
          </div>
          <div class="metric">
            <div class="m-label">30일 변동성</div>
            <div class="m-val">{s["volatility"]:.1f}%</div>
          </div>
          <div class="metric">
            <div class="m-label">PER</div>
            <div class="m-val">{f'{s["pe"]:.1f}x' if s["pe"] else 'N/A'}</div>
          </div>
          <div class="metric">
            <div class="m-label">PBR</div>
            <div class="m-val">{f'{s["pb"]:.2f}x' if s["pb"] else 'N/A'}</div>
          </div>
          <div class="metric">
            <div class="m-label">시가총액</div>
            <div class="m-val">{fcap(s["market_cap"])}</div>
          </div>
          <div class="metric">
            <div class="m-label">배당수익률</div>
            <div class="m-val">{f'{s["dividend_yield"]*100:.2f}%' if s["dividend_yield"] else 'N/A'}</div>
          </div>
        </div>

        {analyst_section(s)}

        <div class="subsection">
          <div class="sub-title">기술 분석 신호</div>
          <div class="sig-list" style="margin-top:10px">
            {signal_items_html(s["signal"])}
          </div>
        </div>
      </div>'''

    sam_card_html = stock_card(sam, sam_sc, sam_sl)
    hyx_card_html = stock_card(hyx, hyx_sc, hyx_sl)

    corr_desc = '높음 — 함께 움직이는 경향 강함' if abs(correlation) > 0.7 else \
                '중간 — 어느 정도 분산 효과 있음' if abs(correlation) > 0.4 else \
                '낮음 — 독립적으로 움직이는 경향'

    def comp_row(label, sv, hv, sv_color='', hv_color=''):
        return f'''
        <div class="comp-row">
          <div class="cv-left" style="color:{sv_color if sv_color else "inherit"}">{sv}</div>
          <div class="cv-mid">{label}</div>
          <div class="cv-right" style="color:{hv_color if hv_color else "inherit"}">{hv}</div>
        </div>'''

    ytd_s_color = '#ef4444' if sam['ytd_return'] > 0 else '#3b82f6'
    ytd_h_color = '#ef4444' if hyx['ytd_return'] > 0 else '#3b82f6'
    chg_s_color = '#ef4444' if sam['change_pct'] > 0 else '#3b82f6'
    chg_h_color = '#ef4444' if hyx['change_pct'] > 0 else '#3b82f6'

    comp_rows = (
        comp_row('당일 등락', fpct(sam['change_pct']), fpct(hyx['change_pct']), chg_s_color, chg_h_color) +
        comp_row('1년 수익률', fpct(sam['ytd_return']), fpct(hyx['ytd_return']), ytd_s_color, ytd_h_color) +
        comp_row('30일 변동성', f"{sam['volatility']:.1f}%", f"{hyx['volatility']:.1f}%") +
        comp_row('PER', f"{sam['pe']:.1f}x" if sam['pe'] else 'N/A', f"{hyx['pe']:.1f}x" if hyx['pe'] else 'N/A') +
        comp_row('PBR', f"{sam['pb']:.2f}x" if sam['pb'] else 'N/A', f"{hyx['pb']:.2f}x" if hyx['pb'] else 'N/A') +
        comp_row('시가총액', fcap(sam['market_cap']), fcap(hyx['market_cap'])) +
        comp_row('목표가(평균)', fp(sam.get('analyst_target')), fp(hyx.get('analyst_target'))) +
        comp_row('투자 신호', sam['signal']['overall'], hyx['signal']['overall'])
    )

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>삼성전자 & SK하이닉스 주식 대시보드</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0a0e1a;--surf:#111827;--surf2:#1f2937;--brd:#374151;
  --txt:#f9fafb;--dim:#9ca3af;--grn:#10b981;--red:#ef4444;--blu:#3b82f6;
}}
body{{font-family:-apple-system,'Segoe UI','Noto Sans KR',sans-serif;background:var(--bg);color:var(--txt);min-height:100vh}}
.header{{background:linear-gradient(135deg,#0f172a,#1e293b);padding:18px 24px;border-bottom:1px solid var(--brd);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px}}
.header h1{{font-size:1.35rem;font-weight:800;letter-spacing:-0.3px}}
.header .sub{{color:var(--dim);font-size:0.8rem;margin-top:3px}}
.upd{{background:var(--surf2);border:1px solid var(--brd);border-radius:8px;padding:7px 14px;font-size:0.78rem;color:var(--dim)}}
.wrap{{max-width:1400px;margin:0 auto;padding:20px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:20px}}
@media(max-width:860px){{.grid2{{grid-template-columns:1fr}}}}
.card{{background:var(--surf);border:1px solid var(--brd);border-radius:14px;padding:22px}}
.stock-header{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:18px}}
.stock-name{{display:flex;align-items:center;gap:10px}}
.stock-dot{{width:11px;height:11px;border-radius:50%;flex-shrink:0;margin-top:3px}}
.stock-title{{font-size:1.15rem;font-weight:700}}
.stock-ticker{{color:var(--dim);font-size:0.8rem;margin-top:2px}}
.badge{{padding:5px 13px;border-radius:16px;font-size:0.78rem;font-weight:600;color:#fff}}
.price-block{{margin-bottom:16px}}
.price-main{{font-size:1.9rem;font-weight:800;letter-spacing:-0.5px}}
.price-change{{margin-top:4px;font-size:0.9rem}}
.price-change.up{{color:var(--red)}}
.price-change.dn{{color:var(--blu)}}
.week52{{margin-bottom:16px}}
.range-track{{position:relative;background:var(--surf2);height:6px;border-radius:3px;margin:8px 0}}
.range-fill{{position:absolute;height:100%;background:linear-gradient(90deg,var(--blu),var(--grn));border-radius:3px}}
.range-pin{{position:absolute;top:-5px;width:16px;height:16px;border-radius:50%;background:#fff;border:2px solid var(--blu);transform:translateX(-50%)}}
.range-labels{{display:flex;justify-content:space-between;font-size:0.73rem;color:var(--dim)}}
.metrics{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px}}
.metric{{background:var(--surf2);border-radius:9px;padding:10px 12px}}
.m-label{{font-size:0.7rem;color:var(--dim);text-transform:uppercase;letter-spacing:0.4px}}
.m-val{{font-size:0.95rem;font-weight:600;margin-top:3px}}
.subsection{{margin-top:16px;padding-top:16px;border-top:1px solid var(--brd)}}
.sub-title{{font-size:0.72rem;text-transform:uppercase;letter-spacing:0.8px;color:var(--dim);font-weight:600;margin-bottom:8px}}
.sig-list{{display:flex;flex-direction:column;gap:5px}}
.sig-item{{display:flex;align-items:center;gap:7px;font-size:0.78rem}}
.sig-dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0}}
.chart-card{{background:var(--surf);border:1px solid var(--brd);border-radius:14px;padding:22px;margin-bottom:20px}}
.chart-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;flex-wrap:wrap;gap:10px}}
.chart-title{{font-size:1rem;font-weight:700}}
.tabs{{display:flex;gap:6px}}
.tab{{padding:6px 16px;border-radius:7px;border:1px solid var(--brd);background:transparent;color:var(--dim);cursor:pointer;font-size:0.82rem;transition:all 0.15s}}
.tab.on{{background:var(--surf2);color:var(--txt);border-color:var(--blu)}}
.chart-wrap{{position:relative}}
.comp-card{{background:var(--surf);border:1px solid var(--brd);border-radius:14px;padding:22px;margin-bottom:20px}}
.comp-title{{font-size:1rem;font-weight:700;margin-bottom:18px}}
.comp-head{{display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:center;padding:8px 0;border-bottom:2px solid var(--brd);margin-bottom:4px}}
.comp-row{{display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:center;padding:9px 0;border-bottom:1px solid var(--surf2)}}
.comp-row:last-child{{border-bottom:none}}
.cv-left{{text-align:right;font-size:0.87rem;font-weight:500}}
.cv-right{{text-align:left;font-size:0.87rem;font-weight:500}}
.cv-mid{{text-align:center;font-size:0.7rem;color:var(--dim);text-transform:uppercase;white-space:nowrap}}
.corr-note{{margin-top:14px;background:var(--surf2);border-radius:9px;padding:12px 16px;font-size:0.82rem;color:var(--dim)}}
.insight-card{{background:var(--surf);border:1px solid var(--brd);border-radius:14px;padding:22px;margin-bottom:20px}}
.insight-title{{font-size:1rem;font-weight:700;margin-bottom:18px}}
.insight-item{{display:flex;gap:14px;padding:12px 0;border-bottom:1px solid var(--surf2)}}
.insight-item:last-child{{border-bottom:none}}
.insight-icon{{font-size:1.2rem;flex-shrink:0;margin-top:1px}}
.i-head{{font-weight:600;margin-bottom:4px;font-size:0.88rem}}
.i-desc{{font-size:0.8rem;color:var(--dim);line-height:1.55}}
.footer{{text-align:center;padding:20px;color:var(--dim);font-size:0.73rem;border-top:1px solid var(--brd)}}
.disc{{font-size:0.7rem;color:#6b7280;margin-top:2px}}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>📈 삼성전자 &amp; SK하이닉스 주식 대시보드</h1>
    <div class="sub">매일 오전 6시(KST) 자동 업데이트 · 1년 예측 포함</div>
  </div>
  <div class="upd">🕐 업데이트: {now.strftime('%Y-%m-%d %H:%M')} KST</div>
</div>

<div class="wrap">

  <!-- 종목 카드 -->
  <div class="grid2">
    {sam_card_html}
    {hyx_card_html}
  </div>

  <!-- 가격 차트 -->
  <div class="chart-card">
    <div class="chart-head">
      <div class="chart-title">📊 주가 차트 (1년 실적 + 1년 예측선)</div>
      <div class="tabs">
        <button class="tab on" onclick="switchStock('samsung')">삼성전자</button>
        <button class="tab" onclick="switchStock('hynix')">SK하이닉스</button>
      </div>
    </div>
    <div class="chart-wrap" style="height:380px"><canvas id="priceChart"></canvas></div>
  </div>

  <!-- 기술적 지표 -->
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

  <!-- 비교 -->
  <div class="comp-card">
    <div class="comp-title">⚖️ 두 종목 비교</div>
    <div class="comp-head">
      <div class="cv-left" style="font-weight:700;color:{sam['color']}">{sam['name']}</div>
      <div class="cv-mid">항목</div>
      <div class="cv-right" style="font-weight:700;color:{hyx['color']}">{hyx['name']}</div>
    </div>
    {comp_rows}
    <div class="corr-note">
      📐 두 종목 상관계수: <strong style="color:var(--txt)">{correlation:.3f}</strong>
      &nbsp;—&nbsp; {corr_desc}
    </div>
  </div>

  <!-- 투자 인사이트 -->
  <div class="insight-card">
    <div class="insight-title">💡 투자 인사이트</div>

    <div class="insight-item">
      <div class="insight-icon">🔮</div>
      <div>
        <div class="i-head">1년 예측 방법론</div>
        <div class="i-desc">과거 6개월 선형 추세와 애널리스트 컨센서스 목표가를 가중 혼합하여 산출합니다. 예측값은 참고용이며 실제 주가와 다를 수 있습니다. 예측 신뢰도는 시장 변동성에 따라 크게 달라집니다.</div>
      </div>
    </div>

    <div class="insight-item">
      <div class="insight-icon">🤝</div>
      <div>
        <div class="i-head">두 종목의 상관관계 ({correlation:.2f})</div>
        <div class="i-desc">모두 메모리 반도체 기업으로 {'높은 동행성을 보입니다. 포트폴리오 분산 효과는 제한적이며' if abs(correlation) > 0.7 else '어느 정도 독립성이 있습니다.'} DRAM/NAND 가격 사이클, HBM 수요, 글로벌 AI 인프라 투자 규모가 두 종목 모두에 큰 영향을 미칩니다.</div>
      </div>
    </div>

    <div class="insight-item">
      <div class="insight-icon">🔵</div>
      <div>
        <div class="i-head">삼성전자 핵심 포인트</div>
        <div class="i-desc">반도체(DS) · 스마트폰(MX) · 디스플레이(SDC) · 가전(DA) 4개 사업부의 복합 구조. HBM3E 공급 확대와 파운드리 GAA 공정 수율이 핵심 변수. 배당 안정성으로 장기 투자에 적합. 메모리 의존도를 낮추는 비메모리 성장 여부 주시.</div>
      </div>
    </div>

    <div class="insight-item">
      <div class="insight-icon">🟠</div>
      <div>
        <div class="i-head">SK하이닉스 핵심 포인트</div>
        <div class="i-desc">HBM 글로벌 1위 공급자로 NVIDIA AI 서버 생태계에 깊이 연결. 순수 메모리 기업 특성상 사이클 변동에 레버리지가 크게 걸림. NAND 사업 흑자 전환 여부와 HBM4 수주 현황이 주요 모멘텀. 고성장·고위험 성격.</div>
      </div>
    </div>

    <div class="insight-item">
      <div class="insight-icon">⚠️</div>
      <div>
        <div class="i-head">주요 리스크 요인</div>
        <div class="i-desc">① 미·중 반도체 수출 규제 강화 ② 글로벌 IT 수요 사이클 둔화 ③ DRAM·NAND 현물가 하락 ④ 원/달러 환율 급변 ⑤ Micron·CXMT 등 경쟁사 공격적 투자 ⑥ AI 버블 우려에 따른 HBM 수요 감소</div>
      </div>
    </div>

    <div class="insight-item">
      <div class="insight-icon">🎯</div>
      <div>
        <div class="i-head">핵심 모니터링 지표</div>
        <div class="i-desc">① DRAM · NAND 현물/고정가 추이 ② HBM 출하량 및 ASP ③ 미국 필라델피아 반도체지수(SOX) ④ AI 서버 업체 Capex 발표(Meta · Microsoft · Google · Amazon) ⑤ 분기 실적 및 가이던스 ⑥ 환율(USD/KRW)</div>
      </div>
    </div>

    <div class="insight-item">
      <div class="insight-icon">📅</div>
      <div>
        <div class="i-head">계절성 및 이벤트 일정</div>
        <div class="i-desc">삼성·하이닉스 실적 발표: 매 분기말 다음달 초. 반도체 주요 컨퍼런스: CES(1월), MWC(2-3월), Computex(5월), Hot Chips(8월). 국내 증시 영향: FOMC 금리 결정, CPI 발표, 코스피 MSCI 비중 조정.</div>
      </div>
    </div>
  </div>

</div>

<div class="footer">
  데이터 출처: Yahoo Finance (yfinance) · 본 대시보드는 투자 참고용이며, 투자 판단의 책임은 본인에게 있습니다.<br>
  <span class="disc">GitHub Actions 자동 생성 · 매일 오전 6시(KST) 업데이트 · 예측은 통계 모델 기반으로 실제 시세와 다를 수 있습니다</span>
</div>

<script>
const DATA = {data_json};
let cur = 'samsung';
let PC, RC, MC, VC;

const BASE_OPT = {{
  responsive: true, maintainAspectRatio: false,
  animation: false,
  plugins: {{
    legend: {{ labels: {{ color: '#9ca3af', font: {{ size: 10 }} }} }},
    tooltip: {{
      mode: 'index', intersect: false,
      backgroundColor: '#1f2937', titleColor: '#f9fafb',
      bodyColor: '#9ca3af', borderColor: '#374151', borderWidth: 1,
    }}
  }},
  scales: {{
    x: {{
      ticks: {{ color: '#6b7280', maxTicksLimit: 9, font: {{ size: 9 }} }},
      grid: {{ color: '#1a2235' }}
    }},
    y: {{
      ticks: {{ color: '#6b7280', font: {{ size: 9 }} }},
      grid: {{ color: '#1a2235' }}
    }}
  }}
}};

function priceTicks(v) {{
  return v != null ? v.toLocaleString('ko-KR') + '원' : '';
}}

function initCharts(key) {{
  [PC, RC, MC, VC].forEach(c => c && c.destroy());
  const d = DATA[key];
  const c = d.chart;

  // Price chart
  PC = new Chart(document.getElementById('priceChart'), {{
    type: 'line',
    data: {{
      labels: c.dates,
      datasets: [
        {{ label: '종가', data: c.prices, borderColor: d.color, borderWidth: 2, pointRadius: 0, fill: false, tension: 0.1, spanGaps: false }},
        {{ label: '1년 예측', data: c.pred, borderColor: d.color_pred, borderWidth: 2, borderDash: [7, 4], pointRadius: 0, fill: false, tension: 0.2, spanGaps: false }},
        {{ label: '20일선', data: c.sma20, borderColor: '#f59e0b', borderWidth: 1, pointRadius: 0, fill: false, tension: 0.1, spanGaps: false }},
        {{ label: '50일선', data: c.sma50, borderColor: '#8b5cf6', borderWidth: 1, pointRadius: 0, fill: false, tension: 0.1, spanGaps: false }},
        {{ label: '200일선', data: c.sma200, borderColor: '#ef4444', borderWidth: 1, pointRadius: 0, fill: false, tension: 0.1, spanGaps: false }},
        {{ label: '볼린저↑', data: c.bb_upper, borderColor: '#4b5563', borderWidth: 1, borderDash: [3, 3], pointRadius: 0, fill: false, spanGaps: false }},
        {{ label: '볼린저↓', data: c.bb_lower, borderColor: '#4b5563', borderWidth: 1, borderDash: [3, 3], pointRadius: 0, fill: false, spanGaps: false }},
      ]
    }},
    options: {{
      ...BASE_OPT,
      plugins: {{
        ...BASE_OPT.plugins,
        tooltip: {{ ...BASE_OPT.plugins.tooltip, callbacks: {{ label: ctx => ctx.raw != null ? ' ' + ctx.dataset.label + ': ' + ctx.raw.toLocaleString('ko-KR') + '원' : null }} }}
      }},
      scales: {{ x: BASE_OPT.scales.x, y: {{ ...BASE_OPT.scales.y, ticks: {{ color: '#6b7280', font: {{ size: 9 }}, callback: priceTicks }} }} }}
    }}
  }});

  // RSI
  RC = new Chart(document.getElementById('rsiChart'), {{
    type: 'line',
    data: {{
      labels: c.dates,
      datasets: [
        {{ label: 'RSI', data: c.rsi, borderColor: '#a78bfa', borderWidth: 1.5, pointRadius: 0, fill: false, spanGaps: false }},
        {{ label: '과매수(70)', data: c.dates.map(() => 70), borderColor: '#ef4444', borderWidth: 1, borderDash: [4, 4], pointRadius: 0, fill: false }},
        {{ label: '과매도(30)', data: c.dates.map(() => 30), borderColor: '#3b82f6', borderWidth: 1, borderDash: [4, 4], pointRadius: 0, fill: false }},
      ]
    }},
    options: {{ ...BASE_OPT, scales: {{ x: BASE_OPT.scales.x, y: {{ ...BASE_OPT.scales.y, min: 0, max: 100 }} }} }}
  }});

  // MACD
  MC = new Chart(document.getElementById('macdChart'), {{
    data: {{
      labels: c.dates,
      datasets: [
        {{ type: 'bar', label: 'MACD 히스토그램', data: c.macd_hist, backgroundColor: c.macd_hist.map(v => v >= 0 ? '#10b98166' : '#ef444466') }},
        {{ type: 'line', label: 'MACD', data: c.macd, borderColor: '#3b82f6', borderWidth: 1.5, pointRadius: 0, fill: false, spanGaps: false }},
        {{ type: 'line', label: '시그널', data: c.macd_signal, borderColor: '#f97316', borderWidth: 1.5, pointRadius: 0, fill: false, spanGaps: false }},
      ]
    }},
    options: BASE_OPT
  }});

  // Volume
  VC = new Chart(document.getElementById('volChart'), {{
    type: 'bar',
    data: {{
      labels: c.dates,
      datasets: [{{ label: '거래량', data: c.volume, backgroundColor: d.color + '66', borderWidth: 0 }}]
    }},
    options: {{
      ...BASE_OPT,
      scales: {{
        x: BASE_OPT.scales.x,
        y: {{ ...BASE_OPT.scales.y, ticks: {{ color: '#6b7280', font: {{ size: 9 }}, callback: v => v >= 1e6 ? (v/1e6).toFixed(0)+'M' : v.toLocaleString() }} }}
      }}
    }}
  }});
}}

function switchStock(key) {{
  cur = key;
  document.querySelectorAll('.tab').forEach((b, i) => b.classList.toggle('on', (key==='samsung'&&i===0)||(key==='hynix'&&i===1)));
  initCharts(key);
}}

initCharts('samsung');
</script>
</body>
</html>"""
    return html


def main():
    print(f"[{datetime.now(KST).strftime('%Y-%m-%d %H:%M')} KST] 대시보드 생성 시작")
    data = {}
    for key in STOCKS:
        result = process_stock(key)
        if result:
            data[key] = result
        else:
            print(f"  {STOCKS[key]['name']} 데이터 로드 실패 — 스킵", file=sys.stderr)

    if len(data) < 2:
        print("데이터 부족으로 생성 중단", file=sys.stderr)
        sys.exit(1)

    html = generate_html(data)
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html)

    print("index.html 생성 완료")
    for k, d in data.items():
        sign = '+' if d['change'] > 0 else ''
        print(f"  {d['name']}: {d['price']:,.0f}원 ({sign}{d['change_pct']:.2f}%) | 신호: {d['signal']['overall']}")


if __name__ == '__main__':
    main()

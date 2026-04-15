import re, os, json, base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yaml
from datetime import datetime
from collections import defaultdict


EUR_TO_USD = 1.2   # €1 = $1.20

def parse_price(s) -> float | None:
    """Parse messy price strings (€/$/USD/EUR, various formats) to USD dollars."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()

    m = re.match(r'€\s*(\d+)[¢](\d+)', s) or re.match(r'(\d+)€(\d+)¢', s)
    if m:
        return round((int(m.group(1)) + int(m.group(2)) / 100) * EUR_TO_USD, 2)


    m = re.match(r'€\s*([\d.]+)', s) or re.match(r'([\d.]+)\s*€', s)
    if m:
        v = m.group(1).rstrip('.')
        return round(float(v) * EUR_TO_USD, 2) if v else None

    m = re.search(r'([\d.]+)\s*EUR|EUR\s*([\d.]+)', s, re.I)
    if m:
        v = (m.group(1) or m.group(2)).rstrip('.')
        return round(float(v) * EUR_TO_USD, 2) if v else None

    m = re.search(r'([\d.]+)\s*\$|\$\s*([\d.]+)', s)
    if m:
        v = (m.group(1) or m.group(2)).rstrip('.')
        return round(float(v), 2) if v else None

    m = re.search(r'USD\s*([\d.]+)|([\d.]+)\s*USD', s, re.I)
    if m:
        v = (m.group(1) or m.group(2)).rstrip('.')
        return round(float(v), 2) if v else None

    # Bare number
    if re.match(r'^[\d.]+$', s):
        return round(float(s), 2)

    return None


_TS_FMTS = [
    '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d',
    '%m/%d/%y %H:%M:%S', '%m/%d/%y %I:%M:%S %p', '%m/%d/%y %H:%M:%S %p',
    '%m/%d/%y %H:%M',    '%m/%d/%y %I:%M %p',
    '%H:%M:%S %Y-%m-%d', '%H:%M %Y-%m-%d',
    '%H:%M:%S %m/%d/%y', '%H:%M %m/%d/%y',
    '%H:%M:%S %p %d-%B-%Y', '%I:%M:%S %p %d-%B-%Y',
    '%H:%M:%S %p %d-%b-%Y', '%I:%M:%S %p %d-%b-%Y',
    '%H:%M %p %d-%B-%Y',    '%I:%M %p %d-%B-%Y',
    '%d-%B-%Y %H:%M:%S', '%d-%b-%Y %H:%M:%S',
    '%d-%B-%Y %H:%M',    '%d-%b-%Y %H:%M',
    '%a %b %d %H:%M:%S %Y',
    '%d-%b-%Y', '%d-%B-%Y',
]

def parse_ts(s) -> datetime | None:
    if not s or not isinstance(s, str):
        return None
    s = re.sub(r'A\.M\.', 'AM', s.strip(), flags=re.I)
    s = re.sub(r'P\.M\.', 'PM', s, flags=re.I)
    sn = re.sub(r'[;,]', ' ', s).strip()
    for fmt in _TS_FMTS:
        try:
            return datetime.strptime(sn, fmt)
        except ValueError:
            pass
    try:
        from dateutil import parser as dp
        return dp.parse(sn)
    except Exception:
        pass
    return None


def dedupe_users(users_df: pd.DataFrame) -> pd.Series:
    """
    Union–Find over users sharing at least one of:
    normalised name, normalised phone, normalised address, email.
    Returns a Series mapping user_id → canonical_id.
    """
    def norm_phone(p):
        return re.sub(r'\D', '', p) if isinstance(p, str) else ''

    def norm_name(n):
        if not isinstance(n, str):
            return ''
        n = re.sub(
            r'\b(Dr|Rep|Sen|Mr|Mrs|Ms|Miss|Prof|Rev|Jr|Sr|II|III|IV|LLD|MD|PhD)\.?\b',
            '', n, flags=re.I)
        return ' '.join(n.lower().split())

    def norm_addr(a):
        return re.sub(r'\s+', ' ', a.lower().strip()) if isinstance(a, str) else ''

    u = users_df.copy()
    u['_name']  = u['name'].apply(norm_name)
    u['_phone'] = u['phone'].apply(norm_phone)
    u['_addr']  = u['address'].apply(norm_addr)
    u['_email'] = u['email'].str.lower().str.strip().fillna('')

    parent = {uid: uid for uid in u['id']}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for field in ['_name', '_phone', '_addr', '_email']:
        idx = defaultdict(list)
        for _, row in u.iterrows():
            if row[field]:
                idx[row[field]].append(row['id'])
        for ids in idx.values():
            for uid in ids[1:]:
                union(ids[0], uid)

    return u['id'].apply(find)



def analyse(ds: int, data_dir: str, out_dir: str) -> dict:
    print(f'\n{"="*55}  Dataset {ds}  {"="*55}')


    orders = pd.read_parquet(os.path.join(data_dir, f'orders.parquet_data-{ds}_'))
    users  = pd.read_csv(os.path.join(data_dir, f'users_data-{ds}_.csv'))
    with open(os.path.join(data_dir, f'books_data-{ds}_.yaml')) as f:
        books_raw = yaml.safe_load(f)
    books = pd.DataFrame([{k.lstrip(':'): v for k, v in b.items()} for b in books_raw])

    print(f'  Loaded  orders={len(orders):,}  users={len(users):,}  books={len(books):,}')

    orders = orders.drop_duplicates()

    orders['unit_price_usd'] = orders['unit_price'].apply(parse_price)
    orders['quantity']       = pd.to_numeric(orders['quantity'], errors='coerce')
    orders = orders.dropna(subset=['unit_price_usd', 'quantity'])
    orders['unit_price_usd'] = orders['unit_price_usd'].round(2)

    orders['paid_price'] = (orders['quantity'] * orders['unit_price_usd']).round(2)

    orders['parsed_ts'] = orders['timestamp'].apply(parse_ts)
    orders = orders.dropna(subset=['parsed_ts'])
    orders['year']  = orders['parsed_ts'].dt.year
    orders['month'] = orders['parsed_ts'].dt.month
    orders['day']   = orders['parsed_ts'].dt.day
    orders['date']  = orders['parsed_ts'].dt.date

    print(f'  After cleaning: {len(orders):,} orders  '
          f'({orders["date"].min()} → {orders["date"].max()})')

    daily = (orders.groupby('date')['paid_price']
             .sum().reset_index().rename(columns={'paid_price': 'revenue'}))
    daily_sorted = daily.sort_values('date')
    top5 = daily.nlargest(5, 'revenue').sort_values('revenue', ascending=False)
    print('\n  Top 5 days by revenue:')
    for _, r in top5.iterrows():
        print(f'    {r.date}  ${r.revenue:,.2f}')

    canonical = dedupe_users(users)
    users = users.copy()
    users['canonical_id'] = canonical.values
    n_unique = users['canonical_id'].nunique()
    print(f'\n  Unique users (after reconciliation): {n_unique:,}')


    def norm_authors(a):
        if not isinstance(a, str):
            return frozenset()
        parts = [re.sub(
            r'\b(Dr|Rep|Sen|Mr|Mrs|Ms|Miss|Prof|Rev|Jr|Sr|II|III|IV|LLD|MD|PhD)\.?\b',
            '', p, flags=re.I).strip().lower()
            for p in re.split(r',', a)]
        return frozenset(p for p in parts if p)

    books = books.copy()
    books['author_set'] = books['author'].apply(norm_authors)
    n_author_sets = books['author_set'].nunique()
    print(f'  Unique author sets: {n_author_sets}')


    merged = orders.merge(
        books[['id', 'author', 'author_set']],
        left_on='book_id', right_on='id', how='left'
    ).dropna(subset=['author_set'])
    author_sales = (merged.groupby('author_set')['quantity']
                    .sum().reset_index().rename(columns={'quantity': 'sold'})
                    .sort_values('sold', ascending=False))
    top_set  = author_sales.iloc[0]['author_set']
    top_sold = int(author_sales.iloc[0]['sold'])
    top_author_name = ', '.join(sorted(top_set))
    print(f'  Most popular author set: {top_author_name!r}  ({top_sold} copies)')

   
    ouid = orders.merge(users[['id', 'canonical_id']],
                        left_on='user_id', right_on='id', how='left')
    spending = (ouid.groupby('canonical_id')['paid_price']
                .sum().reset_index().rename(columns={'paid_price': 'total'})
                .sort_values('total', ascending=False))
    top_canonical = spending.iloc[0]['canonical_id']
    top_spent     = float(spending.iloc[0]['total'])
    alias_ids     = sorted(users[users['canonical_id'] == top_canonical]['id'].tolist())
    print(f'  Top customer: {alias_ids}  (${top_spent:,.2f})')


    dates = [datetime(d.year, d.month, d.day) for d in daily_sorted['date']]
    revs  = daily_sorted['revenue'].tolist()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, revs, linewidth=1.5, color='#2563EB', alpha=0.9)
    ax.fill_between(dates, revs, alpha=0.12, color='#2563EB')
    ax.set_title(f'Daily Revenue — Dataset {ds}', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Date'); ax.set_ylabel('Revenue (USD)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha='right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    chart_path = os.path.join(out_dir, f'revenue_ds{ds}.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'ds': ds,
        'top5_days': [
            {'date': str(r.date), 'revenue': round(float(r.revenue), 2)}
            for _, r in top5.iterrows()
        ],
        'n_unique_users':    n_unique,
        'n_author_sets':     n_author_sets,
        'top_author':        top_author_name,
        'top_customer_ids':  alias_ids,
        'top_customer_spent': round(top_spent, 2),
        'daily_revenue': [
            {'date': str(d), 'revenue': round(float(rev), 2)}
            for d, rev in zip(daily_sorted['date'], daily_sorted['revenue'])
        ],
    }



def build_dashboard(results: dict, out_dir: str):
    charts = {}
    for ds in [1, 2, 3]:
        path = os.path.join(out_dir, f'revenue_ds{ds}.png')
        with open(path, 'rb') as f:
            charts[str(ds)] = base64.b64encode(f.read()).decode()

    def render_panel(ds, r, chart_b64):
        top5   = r['top5_days']
        ids    = r['top_customer_ids']
        ids_str = '[' + ', '.join(str(i) for i in ids) + ']'
        badges = ['gold', 'silver', 'bronze', 'other', 'other']

        rows = ''
        for i, d in enumerate(top5):
            rows += (
                f'<tr>'
                f'<td><span class="rank-badge {badges[i]}">{i+1}</span>'
                f'<span style="font-family:monospace">{d["date"]}</span></td>'
                f'<td>${d["revenue"]:,.2f}</td>'
                f'</tr>'
            )
        chips = ''.join(f'<span class="chip">{i}</span>' for i in ids)

        return f"""
<div id="panel-{ds}" class="panel {'active' if ds==1 else ''}">
  <div class="kpi-grid">
    <div class="kpi blue"><label>Unique Real Users</label>
      <div class="value">{r['n_unique_users']:,}</div>
      <div class="sub">After identity reconciliation</div></div>
    <div class="kpi green"><label>Unique Author Sets</label>
      <div class="value">{r['n_author_sets']:,}</div>
      <div class="sub">Solo &amp; co-author combinations</div></div>
    <div class="kpi purple"><label>Top Customer Spend</label>
      <div class="value">${r['top_customer_spent']:,.2f}</div>
      <div class="sub">IDs: {ids_str}</div></div>
    <div class="kpi yellow"><label>Peak Day Revenue</label>
      <div class="value">${top5[0]['revenue']:,.2f}</div>
      <div class="sub">{top5[0]['date']}</div></div>
  </div>
  <div class="two-col">
    <div class="card">
      <div class="section-title"><span class="dot"></span>Top 5 Days by Revenue</div>
      <table class="rank-table">
        <thead><tr><th>Date</th><th style="text-align:right">Revenue (USD)</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <div class="card">
      <div class="section-title"><span class="dot"></span>Key Metrics</div>
      <div class="info-row"><span class="key">Most Popular Author(s)</span></div>
      <div class="badge-author"><span class="icon">✍️</span>{r['top_author'].title()}</div>
      <div style="margin-top:18px">
        <div class="info-row">
          <span class="key">Top Buyer — All IDs</span>
          <div class="ids-chips">{chips}</div>
        </div>
        <div class="info-row">
          <span class="key">Total Spent</span>
          <span class="val" style="color:#10b981">${r['top_customer_spent']:,.2f}</span>
        </div>
        <div class="info-row">
          <span class="key">Unique Real Users</span>
          <span class="val">{r['n_unique_users']:,}</span>
        </div>
        <div class="info-row">
          <span class="key">Unique Author Sets</span>
          <span class="val">{r['n_author_sets']:,}</span>
        </div>
      </div>
    </div>
  </div>
  <div class="card">
    <div class="section-title"><span class="dot"></span>Daily Revenue Chart — Dataset {ds}</div>
    <div class="chart-wrap">
      <img src="data:image/png;base64,{chart_b64}" alt="Daily Revenue DS{ds}">
    </div>
  </div>
</div>"""

    CSS = """
:root{--bg:#0f172a;--surface:#1e293b;--surface2:#273548;--border:#334155;
  --accent:#3b82f6;--accent2:#6366f1;--green:#10b981;--yellow:#f59e0b;
  --text:#f1f5f9;--muted:#94a3b8;--radius:12px}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;min-height:100vh}
header{background:linear-gradient(135deg,#1e3a5f,#1e293b);border-bottom:1px solid var(--border);
  padding:20px 40px;display:flex;align-items:center;gap:16px}
header .logo{font-size:28px}
header h1{font-size:22px;font-weight:700;letter-spacing:-.5px}
header p{font-size:13px;color:var(--muted);margin-top:2px}
.tabs{display:flex;gap:4px;padding:20px 40px 0;border-bottom:1px solid var(--border);background:var(--surface)}
.tab{padding:10px 24px;border-radius:8px 8px 0 0;cursor:pointer;font-size:14px;font-weight:600;
  background:transparent;color:var(--muted);border:1px solid transparent;border-bottom:none;transition:all .2s}
.tab:hover{color:var(--text);background:var(--surface2)}
.tab.active{background:var(--bg);color:var(--accent);border-color:var(--border);border-bottom-color:var(--bg);margin-bottom:-1px}
.panel{display:none;padding:32px 40px}.panel.active{display:block}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-bottom:28px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  padding:20px 24px;position:relative;overflow:hidden}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:3px}
.kpi.blue::before{background:var(--accent)}.kpi.green::before{background:var(--green)}
.kpi.purple::before{background:var(--accent2)}.kpi.yellow::before{background:var(--yellow)}
.kpi label{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;color:var(--muted)}
.kpi .value{font-size:28px;font-weight:700;margin:8px 0 4px;letter-spacing:-1px}
.kpi .sub{font-size:12px;color:var(--muted)}
.section-title{font-size:16px;font-weight:700;margin-bottom:14px;padding-bottom:10px;
  border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px}
.section-title .dot{width:8px;height:8px;border-radius:50%;background:var(--accent);display:inline-block}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px}
@media(max-width:900px){.two-col{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px 24px}
.rank-table{width:100%;border-collapse:collapse;font-size:14px}
.rank-table th{text-align:left;padding:8px 12px;font-size:11px;font-weight:600;text-transform:uppercase;
  letter-spacing:.6px;color:var(--muted);border-bottom:1px solid var(--border)}
.rank-table th:last-child{text-align:right}
.rank-table td{padding:10px 12px;border-bottom:1px solid #1e2d40}
.rank-table td:last-child{text-align:right;font-weight:600;color:var(--green);font-size:15px}
.rank-table tr:last-child td{border-bottom:none}
.rank-badge{display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;
  border-radius:50%;font-size:11px;font-weight:700;margin-right:8px}
.rank-badge.gold{background:#78350f;color:#fbbf24}.rank-badge.silver{background:#1f2937;color:#9ca3af}
.rank-badge.bronze{background:#1c1a17;color:#b45309}.rank-badge.other{background:#1e2d40;color:var(--muted)}
.info-row{display:flex;justify-content:space-between;align-items:center;
  padding:10px 0;border-bottom:1px solid #1e2d40;font-size:14px}
.info-row:last-child{border-bottom:none}
.info-row .key{color:var(--muted);font-size:13px}
.info-row .val{font-weight:600;text-align:right;max-width:60%}
.ids-chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px;justify-content:flex-end}
.chip{background:#1e3a5f;border:1px solid #2d5c8a;color:#60a5fa;border-radius:20px;
  padding:3px 10px;font-size:12px;font-weight:600;font-family:monospace}
.chart-wrap{margin-top:8px}.chart-wrap img{width:100%;border-radius:8px;display:block}
.badge-author{background:linear-gradient(135deg,#312e81,#1e3a5f);border:1px solid #4c1d95;
  border-radius:8px;padding:12px 16px;font-size:15px;font-weight:700;color:#a78bfa;
  display:flex;align-items:center;gap:10px;margin-top:6px}
.badge-author .icon{font-size:20px}
footer{text-align:center;padding:20px;color:var(--muted);font-size:12px;
  border-top:1px solid var(--border);margin-top:20px}
"""
    HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bookstore Analytics Dashboard</title>
<style>{CSS}</style></head><body>
<header>
  <div class="logo">📚</div>
  <div><h1>Bookstore Analytics Dashboard</h1>
  <p>Sales intelligence across 3 datasets · Prices normalised to USD · Users de-duplicated by identity reconciliation</p></div>
</header>
<div class="tabs">
  <div class="tab active" onclick="switchTab(1)">Dataset 1</div>
  <div class="tab" onclick="switchTab(2)">Dataset 2</div>
  <div class="tab" onclick="switchTab(3)">Dataset 3</div>
</div>
{''.join(render_panel(ds, results[str(ds)], charts[str(ds)]) for ds in [1,2,3])}
<footer>Bookstore Analytics · Prices in USD (€1 = $1.20) · Users reconciled by shared name / phone / email / address</footer>
<script>
function switchTab(n){{
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',i===n-1));
  document.querySelectorAll('.panel').forEach((p,i)=>p.classList.toggle('active',i===n-1));
}}
</script></body></html>"""

    out = os.path.join(out_dir, 'dashboard.html')
    with open(out, 'w') as f:
        f.write(HTML)
    print(f'\n  Dashboard: {out}  ({len(HTML):,} bytes)')



if __name__ == '__main__':
    DATA_DIR = '.'  
    OUT_DIR  = '.'       

    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = {}
    for ds in [1, 2, 3]:
        all_results[ds] = analyse(ds, DATA_DIR, OUT_DIR)

    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print('\nresults.json saved.')

    build_dashboard(all_results, OUT_DIR)
    print('\nDone ✓')

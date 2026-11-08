"""
generate_data.py
Creates a SYNTHETIC, intentionally messy Archtics-style season ticket dataset,
modeled on New York Knicks home games at Madison Square Garden (2025-26).
Members are mostly from New York, then New Jersey, Pennsylvania and Vermont.

Outputs (in data/raw/):
  ticket_activity_2025_26.csv  - exactly 25,000 rows, one row per account per event
  renewals_2026_27.csv         - one row per account: did they renew for next season?

Nothing here is real customer data. Names, emails and locations come from Faker.
The messiness (mixed case, blanks, bad ages, odd date formats, duplicates, empty
columns) is on purpose, so the project includes a real data-cleaning step.

Run:  python src/generate_data.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
TARGET_ROWS = 25_000
rng = np.random.default_rng(SEED)
fake = Faker("en_US")
Faker.seed(SEED)

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Event calendar: 41 home games + 8 arena concerts, 2025-26 season
# ----------------------------------------------------------------------------
OPPONENTS = ["Celtics", "Nets", "76ers", "Raptors", "Bulls", "Cavaliers", "Pistons", "Pacers",
             "Bucks", "Hawks", "Hornets", "Heat", "Magic", "Wizards", "Nuggets", "Timberwolves",
             "Thunder", "Trail Blazers", "Jazz", "Warriors", "Clippers", "Lakers", "Suns", "Kings",
             "Mavericks", "Rockets", "Grizzlies", "Pelicans", "Spurs"]

game_dates = pd.to_datetime(sorted(rng.choice(
    pd.date_range("2025-10-22", "2026-04-12", freq="D"), size=41, replace=False)))
games = pd.DataFrame({
    "event_name": [f"Knicks Home Game {i+1:02d} vs {rng.choice(OPPONENTS)}" for i in range(41)],
    "event_type": "Game",
    "event_date": game_dates,
    "game_num": range(1, 42),
})

CONCERT_ACTS = ["The Copper Lanterns", "Hudson Line Revival", "Neon Harbor", "Sadie Mae Collins",
                "Midnight Interstate", "The Bowery Saints", "Echo Valley Tour", "Lena Ward Live"]
concert_dates = pd.to_datetime(sorted(rng.choice(
    pd.date_range("2025-10-01", "2026-04-30", freq="D"), size=8, replace=False)))
concerts = pd.DataFrame({
    "event_name": [f"Concert - {a}" for a in CONCERT_ACTS],
    "event_type": "Concert",
    "event_date": concert_dates,
    "game_num": np.nan,
})

# ----------------------------------------------------------------------------
# 2. Price codes: Bowl = letters A-Z (A most expensive), Balcony = 17-30
# ----------------------------------------------------------------------------
BOWL_LEVELS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
BALCONY_LEVELS = [str(n) for n in range(17, 31)]
bowl_price = dict(zip(BOWL_LEVELS, np.linspace(650, 140, len(BOWL_LEVELS)).round()))
balc_price = dict(zip(BALCONY_LEVELS, np.linspace(165, 65, len(BALCONY_LEVELS)).round()))

PLAN_SUFFIX = {"Full Season": "_F", "Half": "_H", "Quarter": "_Q"}
PLAN_MARKUP = {"Full Season": 1.00, "Half": 1.08, "Quarter": 1.15}  # smaller plans pay more per game

# (city, state, ZIP prefix, approx. miles from Madison Square Garden)
NYC = [("New York", "NY", "100", 3), ("Brooklyn", "NY", "112", 8), ("Queens", "NY", "113", 10),
       ("Bronx", "NY", "104", 10), ("Staten Island", "NY", "103", 16)]
NY_SUBURBS = [("Yonkers", "NY", "107", 16), ("White Plains", "NY", "106", 27), ("New Rochelle", "NY", "108", 19),
              ("Scarsdale", "NY", "105", 22), ("Garden City", "NY", "115", 22), ("Hempstead", "NY", "115", 24),
              ("Huntington", "NY", "117", 38), ("Massapequa", "NY", "117", 33)]
NJ = [("Hoboken", "NJ", "070", 3), ("Jersey City", "NJ", "073", 5), ("Newark", "NJ", "071", 12),
      ("Montclair", "NJ", "070", 15), ("Hackensack", "NJ", "076", 13), ("Paramus", "NJ", "076", 18),
      ("Fort Lee", "NJ", "070", 9), ("Morristown", "NJ", "079", 31), ("Edison", "NJ", "088", 30),
      ("Princeton", "NJ", "085", 50)]
UPSTATE_PA_VT = [("Poughkeepsie", "NY", "126", 80), ("Albany", "NY", "122", 150), ("Syracuse", "NY", "132", 250),
                 ("Rochester", "NY", "146", 335), ("Philadelphia", "PA", "191", 95), ("Allentown", "PA", "181", 90),
                 ("Scranton", "PA", "185", 120), ("Pittsburgh", "PA", "152", 370),
                 ("Burlington", "VT", "054", 300), ("Brattleboro", "VT", "053", 190)]


def make_location():
    r = rng.random()
    group = NYC if r < 0.55 else NY_SUBURBS if r < 0.70 else NJ if r < 0.88 else UPSTATE_PA_VT
    city, st, z, miles = group[rng.integers(len(group))]
    miles = miles * rng.uniform(0.7, 1.3)
    zipc = f"{z}{rng.integers(1, 100):02d}"
    return city, st, zipc, max(miles, 1)


# ----------------------------------------------------------------------------
# 3. Accounts (each with hidden "true" behavior traits that drive renewal)
# ----------------------------------------------------------------------------
def make_account(i):
    acct_type = "Corporate" if rng.random() < 0.15 else "Individual"
    first, last = fake.first_name(), fake.last_name()
    city, st, zipc, miles = make_location()
    age = int(np.clip(rng.normal(46, 13), 21, 84))
    member_since = int(np.clip(2025 - rng.exponential(5), 2004, 2025))
    plan = rng.choice(["Full Season", "Half", "Quarter"], p=[0.38, 0.34, 0.28])
    if acct_type == "Corporate":
        plan = rng.choice(["Full Season", "Half"], p=[0.75, 0.25])

    bowl = rng.random() < (0.75 if acct_type == "Corporate" else 0.58)
    level = (BOWL_LEVELS[min(int(rng.beta(1.6, 2.4) * 26), 25)] if bowl
             else BALCONY_LEVELS[rng.integers(len(BALCONY_LEVELS))])
    base_price = bowl_price[level] if bowl else balc_price[level]
    price = round(base_price * PLAN_MARKUP[plan], 2)

    num_seats = int(rng.choice([1, 2, 2, 2, 3, 4, 4, 6] if acct_type == "Individual" else [2, 4, 4, 6, 8]))
    section = int(rng.integers(101, 121) if bowl else rng.integers(201, 228))   # MSG 100s / 200s
    row = (chr(ord("A") + rng.integers(0, 20)) if bowl else str(rng.integers(1, 16)))
    first_seat = int(rng.integers(1, 20))
    seat_str = f"{first_seat}-{first_seat + num_seats - 1}" if num_seats > 1 else str(first_seat)

    # Payment plan: cheaper seats / younger buyers lean toward installments
    p_full = np.clip(0.35 + 0.004 * (age - 40) + 0.0005 * (price - 250) + (0.3 if acct_type == "Corporate" else 0), 0.05, 0.92)
    rest = 1 - p_full
    pay_plan = rng.choice(["Pay in Full", "3 Way Payment Plan", "5-6 Month Payment Plan", "12 Payments, Monthly"],
                          p=[p_full, rest * 0.30, rest * 0.30, rest * 0.40])

    # Hidden traits (not exported directly)
    tenure = 2025 - member_since
    engagement = (0.6 + 0.04 * min(tenure, 12) - 0.0025 * miles
                  - (0.25 if acct_type == "Corporate" else 0) + rng.normal(0, 0.25))
    resale_tendency = np.clip(0.04 + 0.0006 * miles + (0.10 if acct_type == "Corporate" else 0)
                              - 0.15 * engagement + rng.normal(0, 0.05), 0, 0.6)
    pay_risk = np.clip(0.03 + {"Pay in Full": 0, "3 Way Payment Plan": 0.04,
                               "5-6 Month Payment Plan": 0.07, "12 Payments, Monthly": 0.10}[pay_plan]
                       - 0.01 * min(tenure, 10) + rng.normal(0, 0.04), 0, 0.5)

    return dict(account_id=f"ACCT{100000 + i}", first=first, last=last,
                email=f"{first.lower()}.{last.lower()}{rng.integers(1, 99) if rng.random() < 0.4 else ''}@"
                      f"{rng.choice(['gmail.com', 'yahoo.com', 'icloud.com', 'outlook.com', 'aol.com', 'verizon.net', 'optonline.net'], p=[.45, .15, .15, .11, .04, .05, .05])}"
                      if acct_type == "Individual" else f"{first[0].lower()}{last.lower()}@{fake.domain_name()}",
                age=age, city=city, state=st, zip=zipc, miles=miles, account_type=acct_type,
                member_since=member_since, plan=plan, bowl=bowl, level=level, price=price,
                num_seats=num_seats, section=section, row=row, seats=seat_str, pay_plan=pay_plan,
                engagement=engagement, resale_tendency=resale_tendency, pay_risk=pay_risk,
                concert_buyer=rng.random() < 0.22)


def account_events(a):
    if a["plan"] == "Full Season":
        g = games
    elif a["plan"] == "Half":
        g = games[games.game_num % 2 == rng.integers(0, 2)]          # Plan A / Plan B
    else:
        g = games.sample(10, random_state=int(rng.integers(1e9)))      # 10-game pack
    ev = [g]
    if a["concert_buyer"]:
        ev.append(concerts.sample(int(rng.integers(1, 4)), random_state=int(rng.integers(1e9))))
    return pd.concat(ev).sort_values("event_date")


def seat_and_scan(a, is_concert):
    """Seat status codes: RE resold, P posted, F forwarded, O out for forward, D donated, '' = kept."""
    eng = a["engagement"] + (0.3 if is_concert else 0)
    p_re = a["resale_tendency"] * 0.65
    p_p = a["resale_tendency"] * 0.35
    p_f = 0.06 + (0.12 if a["account_type"] == "Corporate" else 0)
    p_o = 0.015
    p_d = 0.01 + (0.02 if a["account_type"] == "Corporate" else 0)
    p_keep = max(1 - (p_re + p_p + p_f + p_o + p_d), 0.05)
    probs = np.array([p_re, p_p, p_f, p_o, p_d, p_keep])
    status = rng.choice(["RE", "P", "F", "O", "D", ""], p=probs / probs.sum())

    n = a["num_seats"]
    if status == "RE":
        scanned = rng.binomial(n, 0.93)                    # buyer shows up
    elif status in ("P", "O"):
        scanned = rng.binomial(n, 0.15)                    # listing/forward never completed
    elif status in ("F", "D"):
        scanned = rng.binomial(n, 0.75)
    else:
        show_prob = float(np.clip(0.55 + 0.35 * eng, 0.05, 0.99))
        scanned = rng.binomial(n, show_prob)               # no-shows are a churn signal
    resale = round(a["price"] * rng.uniform(0.55, 1.6), 2) if status == "RE" else np.nan
    return status, scanned, resale


def payment_status(a):
    """ACCOUNT-level billing standing on the season budget (same on every row of the account).
    Green = paid in full, Yellow = partially paid / behind on plan, Red = has not paid."""
    r = rng.random()
    if a["pay_plan"] == "Pay in Full":
        return "Red" if r < 0.02 else "Green"
    if r < a["pay_risk"]:
        return "Red"
    if r < a["pay_risk"] * 3 + 0.06:
        return "Yellow"
    return "Green"


# Build accounts until we have enough rows
rows, accounts = [], []
i = 0
while len(rows) < TARGET_ROWS:
    a = make_account(i)
    a["pay_status"] = payment_status(a)
    accounts.append(a)
    for _, e in account_events(a).iterrows():
        status, scanned, resale = seat_and_scan(a, e.event_type == "Concert")
        rows.append({**a, "event_name": e.event_name, "event_type": e.event_type,
                     "event_date": e.event_date, "seat_status": status,
                     "scanned": scanned, "resale_price": resale})
    i += 1

df = pd.DataFrame(rows)
acct = pd.DataFrame(accounts)

# ----------------------------------------------------------------------------
# 4. Renewal outcome (hidden logic - try to rediscover it with your models!)
# ----------------------------------------------------------------------------
agg = df.groupby("account_id").agg(
    no_show=("scanned", lambda s: 1 - s.sum() / (len(s) * df.loc[s.index, "num_seats"].iloc[0])),
    resale=("seat_status", lambda s: s.isin(["RE", "P"]).mean()))
acct = acct.merge(agg, left_on="account_id", right_index=True)
logit = (2.6 - 3.2 * acct.no_show - 2.0 * acct.resale
         + acct.pay_status.map({"Green": 0.0, "Yellow": -0.9, "Red": -2.3})
         + 0.09 * np.minimum(2025 - acct.member_since, 12) - 0.0018 * acct.miles
         + acct.plan.map({"Full Season": 0.3, "Half": 0.0, "Quarter": -0.4})
         + (acct.account_type == "Corporate") * -0.3 + rng.normal(0, 0.6, len(acct)))
acct["renewed"] = rng.random(len(acct)) < 1 / (1 + np.exp(-logit))

# ----------------------------------------------------------------------------
# 5. Add realistic mess
# ----------------------------------------------------------------------------
def pick(p):
    return rng.random(len(df)) < p


def name_variant(first, last):
    r = rng.random()
    if r < 0.70: return f"{first} {last}"
    if r < 0.80: return f"{first} {last}".upper()
    if r < 0.87: return f"{first} {last}".lower()
    if r < 0.93: return f"{last}, {first}"
    if r < 0.97: return f"  {first}  {last} "
    return ""


# Most fields are consistent per account, with occasional row-level drift
acct_name = {r.account_id: name_variant(r.first, r.last) for r in acct.itertuples()}
df["Name"] = df.account_id.map(acct_name)
drift = pick(0.04)
df.loc[drift, "Name"] = [name_variant(f, l) for f, l in zip(df.loc[drift, "first"], df.loc[drift, "last"])]

email = df.email.copy()
email[pick(0.05)] = ""
email[pick(0.04)] = email.str.upper()
typo = pick(0.02)
email[typo] = email[typo].str.replace("gmail.com", "gmial.com").str.replace("@", "", n=1)
email[pick(0.02)] = email + " "
df["Email"] = email


def age_variant(age):
    r = rng.random()
    if r < 0.78: return str(age)
    if r < 0.84: return f"{age}.0"
    if r < 0.89: return ""
    if r < 0.92: return rng.choice(["unknown", "N/A", "n/a", "-"])
    if r < 0.95: return f" {age} "
    if r < 0.98: return str(rng.choice([0, 999, 120, 2, -1]))
    return f"{age} yrs"


acct_age = {r.account_id: age_variant(r.age) for r in acct.itertuples()}
df["Age"] = df.account_id.map(acct_age)


def loc_variant(city, st, z):
    r = rng.random()
    full_state = {"NY": "New York", "NJ": "New Jersey", "PA": "Pennsylvania", "VT": "Vermont"}[st]
    if city == "New York" and r < 0.15: return rng.choice(["NYC", "New York City", "Manhattan, NY", "Manhattan"])
    if r < 0.55: return f"{city}, {st}"
    if r < 0.65: return f"{city.lower()} {st.lower()}"
    if r < 0.73: return f"{city}, {full_state}"
    if r < 0.80: return z
    if r < 0.86: return f"{city.upper()}, {st}"
    if r < 0.92: return f"{city}, {st} {z}"
    return ""


acct_loc = {r.account_id: loc_variant(r.city, r.state, r.zip) for r in acct.itertuples()}
df["Location"] = df.account_id.map(acct_loc)

df["Account_Type"] = df.account_type
df.loc[pick(0.03), "Account_Type"] = df.account_type.str.upper()
df.loc[pick(0.06), "Account_Type"] = ""

df["Member_Since"] = df.member_since.astype(str)
df.loc[pick(0.04), "Member_Since"] = ""

ev_name = df.event_name.copy()
ev_name[pick(0.03)] = ev_name.str.upper()
ev_name[pick(0.02)] = ev_name.str.replace("Knicks Home Game ", "NYK HG", regex=False)
df["Event_Name"] = ev_name
df["Event_Type"] = df.event_type
df.loc[pick(0.04), "Event_Type"] = df.event_type.str.upper()
df.loc[pick(0.03), "Event_Type"] = ""

fmt = rng.choice(["%Y-%m-%d", "%m/%d/%Y", "%b %d, %Y", "%m/%d/%y"], p=[0.6, 0.25, 0.1, 0.05], size=len(df))
df["Event_Date"] = [d.strftime(f) for d, f in zip(df.event_date, fmt)]
df.loc[pick(0.01), "Event_Date"] = ""

df["Section"] = df.section.astype(str)
df["Row"] = df.row
df["Seat_Numbers"] = df.seats
df["Num_Seats"] = df.num_seats.astype(str)
df.loc[pick(0.01), "Num_Seats"] = ""

code = df.level + df.plan.map(PLAN_SUFFIX)
code[pick(0.04)] = code.str.lower()
code[pick(0.02)] = code + " "
code[pick(0.01)] = code.str.replace("_", "-", regex=False)
df["Price_Code"] = code
df["Ticket_Type"] = df.plan
mismatch = pick(0.01)
df.loc[mismatch, "Ticket_Type"] = rng.choice(["Full Season", "Half", "Quarter"], size=mismatch.sum())
df.loc[pick(0.05), "Ticket_Type"] = ""

price_fmt = rng.random(len(df))
df["Price_Per_Seat"] = np.where(price_fmt < 0.6, df.price.map(lambda p: f"${p:,.2f}"),
                       np.where(price_fmt < 0.9, df.price.astype(str),
                       np.where(price_fmt < 0.95, df.price.map(lambda p: f"{p:.0f}"), "")))

plan_variants = {"Pay in Full": ["Pay in Full", "PIF", "Pay In Full", "pay in full"],
                 "3 Way Payment Plan": ["3 Way Payment Plan", "3-Pay", "3 way pmt plan"],
                 "5-6 Month Payment Plan": ["5-6 Month Payment Plan", "5-6 Mo Plan", "6 Month Plan"],
                 "12 Payments, Monthly": ["12 Payments, Monthly", "12 Monthly", "Monthly (12)"]}
df["Payment_Plan"] = [v[0] if rng.random() < 0.8 else rng.choice(v) for v in df.pay_plan.map(plan_variants)]
df.loc[pick(0.03), "Payment_Plan"] = ""

color_variants = {"Red": ["Red", "RED", "red", " Red"], "Yellow": ["Yellow", "YELLOW", "Yel", "yellow"],
                  "Green": ["Green", "GREEN", "Grn", "green "]}
df["Payment_Status"] = [v[0] if rng.random() < 0.85 else rng.choice(v) for v in df.pay_status.map(color_variants)]
df.loc[pick(0.02), "Payment_Status"] = ""

seat = df.seat_status.copy()
seat[pick(0.03) & (seat != "")] = seat.str.lower()
blank_variant = pick(0.08) & (seat == "")
seat[blank_variant] = rng.choice(["N/A", "-", " "], size=blank_variant.sum())
df["Seat_Status"] = seat

df["Tickets_Scanned"] = df.scanned.astype(str)
df.loc[pick(0.02), "Tickets_Scanned"] = ""
df["Resale_Price"] = df.resale_price.map(lambda p: "" if pd.isna(p) else (f"${p:.2f}" if rng.random() < 0.5 else f"{p:.2f}"))

df["Rep_Code"] = ""
has_rep = pick(0.25)
df.loc[has_rep, "Rep_Code"] = rng.choice(["JM", "KT", "AR", "BW", "SL", "jm"], size=has_rep.sum())
NOTES = ["called re: parking", "wants seat upgrade", "VIP", "asked about payment date", "ADA request",
         "left voicemail", "do not call", "upgrade candidate", "complaint - concessions", "refund requested"]
df["Notes"] = ""
has_note = pick(0.02)
df.loc[has_note, "Notes"] = rng.choice(NOTES, size=has_note.sum())
df["Unnamed: 23"] = ""                                  # fully blank export column

out_cols = ["Account_ID", "Name", "Email", "Age", "Location", "Account_Type", "Member_Since",
            "Event_Name", "Event_Type", "Event_Date", "Section", "Row", "Seat_Numbers", "Num_Seats",
            "Price_Code", "Ticket_Type", "Price_Per_Seat", "Payment_Plan", "Payment_Status",
            "Seat_Status", "Tickets_Scanned", "Resale_Price", "Rep_Code", "Notes", "Unnamed: 23"]
df["Account_ID"] = df.account_id
out = df[out_cols]

# ~1% duplicate rows (re-export glitch), then trim to exactly 25,000 rows
dupes = out.sample(frac=0.01, random_state=SEED)
out = pd.concat([out, dupes]).sort_index(kind="stable")
out = out.iloc[:TARGET_ROWS].reset_index(drop=True)
out.to_csv(OUT_DIR / "ticket_activity_2025_26.csv", index=False)

# ----------------------------------------------------------------------------
# 6. Renewal file (a separate system, so it's messy too)
# ----------------------------------------------------------------------------
kept = acct[acct.account_id.isin(out.Account_ID)].copy()
yes = ["Renewed", "RENEWED", "Y", "renewed", "Yes"]
no = ["Cancelled", "Canceled", "N", "CANCELLED", "No"]
kept["Renewal_Status"] = [rng.choice(yes, p=[.7, .1, .1, .05, .05]) if r else rng.choice(no, p=[.6, .15, .1, .1, .05])
                          for r in kept.renewed]
REASONS = ["Price increase", "Moving away", "Not attending enough", "Team performance",
           "Financial", "Company budget cut", "Bought elsewhere", ""]
kept["Cancellation_Reason"] = [rng.choice(REASONS) if not r else "" for r in kept.renewed]
kept["Processed_Date"] = [d.strftime("%m/%d/%Y") for d in
                          pd.to_datetime("2026-05-01") + pd.to_timedelta(rng.integers(0, 45, len(kept)), unit="D")]
kept["Account_ID"] = kept.account_id
kept.loc[rng.random(len(kept)) < 0.02, "Account_ID"] = kept.account_id.str.lower()
renewals = kept.sample(frac=0.97, random_state=SEED)   # ~3% of accounts missing from this system
renewals[["Account_ID", "Renewal_Status", "Cancellation_Reason", "Processed_Date"]].to_csv(
    OUT_DIR / "renewals_2026_27.csv", index=False)

print(f"ticket_activity: {len(out):,} rows, {out.Account_ID.nunique():,} accounts")
print(f"renewals: {len(renewals):,} rows, renewal rate {kept.renewed.mean():.1%}")

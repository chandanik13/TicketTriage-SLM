"""
TicketTriage-SLM — Synthetic Dataset Generator

Generates ~5,000 realistic customer-support tickets with controlled
variation in language, tone, and structure.  Every ticket gets a
logically consistent set of labels (category, priority, sentiment,
language, resolution).

Usage:
    python -m src.data_generation

Output:
    data/raw/tickets.csv

Design principles:
    - Compositional text: pieces are combined, not copied from templates
    - No label leakage: category keywords are avoided in ticket text
    - Varied tone: polite, neutral, frustrated, formal, casual
    - Controlled noise: minor typos, punctuation variation
    - Reproducible: seeded with RANDOM_SEED (42)
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    CATEGORIES,
    CATEGORY_RESOLUTION_MAP,
    LANGUAGES,
    LOG_FORMAT,
    LOG_LEVEL,
    NUM_SAMPLES,
    PRIORITIES,
    RANDOM_SEED,
    RAW_DATA_DIR,
    RAW_TICKETS_PATH,
    REQUIRED_COLUMNS,
    RESOLUTIONS,
    SENTIMENTS,
    VALID_LABELS,
)

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# TEXT BUILDING BLOCKS
# ══════════════════════════════════════════════
# Each category has multiple *independent* pools of fragments.
# A ticket is assembled by picking one piece from each pool and
# combining them, producing thousands of unique combinations.
# The pools deliberately avoid using the category name itself
# (e.g. the word "billing" never appears in billing templates).

# ──────────────────────────────────────────────
# BILLING templates
# ──────────────────────────────────────────────
BILLING_OPENINGS = [
    "I noticed an extra charge on my credit card statement.",
    "My bank shows two identical transactions from your company.",
    "I was charged twice for the same purchase this month.",
    "There are duplicate charges showing on my card.",
    "I just checked my statement and there's a double payment.",
    "Looks like the payment went through two times.",
    "My card was debited twice for a single order.",
    "I see an unauthorized duplicate deduction on my account.",
    "The same amount was taken from my card twice.",
    "I received two charge notifications for one transaction.",
    "Your system appears to have processed my payment twice.",
    "Could you check my recent transactions? There seem to be two identical ones.",
    "I'm seeing a repeated charge that shouldn't be there.",
    "The payment for my last order appears duplicated.",
    "My credit card was billed double for my recent purchase.",
    "I've been charged an extra amount that I didn't authorize.",
    "There's a charge I don't recognize that matches my last payment exactly.",
    "Two payments of the same amount showed up on my bank statement today.",
    "I think the system charged me twice when I placed my order.",
    "Something went wrong with my payment — the amount was deducted two times.",
]

BILLING_DETAILS = [
    "The amount is ${amount} each time.",
    "Each charge is for ${amount}.",
    "Both transactions show ${amount}.",
    "The duplicate amount is ${amount}.",
    "I was expecting only one charge of ${amount}.",
    "My monthly payment should be ${amount}, not double that.",
    "This happened on {date}.",
    "I noticed this when I checked my statement on {date}.",
    "The transaction date shows {date}.",
    "It appeared on my statement dated {date}.",
    "Order number {order_id} seems to be the one affected.",
    "This is related to order {order_id}.",
    "Reference number: {order_id}.",
    "",
    "",
    "",
]

BILLING_CLOSINGS = [
    "Please reverse the duplicate charge.",
    "I would appreciate it if you could refund the extra amount.",
    "Could you look into this and issue a refund?",
    "I need the extra payment reversed as soon as possible.",
    "Kindly process a refund for the duplicate transaction.",
    "I'd like to get my money back for the extra charge.",
    "Please fix this and credit my account.",
    "Can you resolve this? I shouldn't have been charged twice.",
    "I expect a refund for the duplicate amount.",
    "Would you be able to reverse the second charge?",
    "Please investigate and process a correction.",
    "I need this resolved quickly, please.",
    "How do I get the extra amount refunded?",
    "This needs to be corrected immediately.",
    "I want the overcharge refunded to my card.",
    "",
    "",
]

# ──────────────────────────────────────────────
# TECHNICAL templates
# ──────────────────────────────────────────────
TECHNICAL_OPENINGS = [
    "The app keeps crashing every time I try to open it.",
    "I'm getting a weird error message when I use the application.",
    "The software freezes whenever I try to save my work.",
    "My device won't connect to your service properly.",
    "The page loads very slowly and sometimes times out completely.",
    "I keep getting a connection error when I try to use the app.",
    "The application is not responding after the latest update.",
    "Something is broken in the app — features that used to work don't anymore.",
    "The system keeps showing error code 503 when I try to access my data.",
    "I can't get the app to work on my phone since yesterday.",
    "Your website keeps displaying a blank screen on my browser.",
    "The download keeps failing at 80 percent every single time.",
    "After the recent update, the app doesn't load properly.",
    "I'm experiencing constant buffering and lag on your platform.",
    "The search function in your app returns no results even though I know there should be matches.",
    "The app is super slow and basically unusable right now.",
    "My data isn't syncing across devices anymore.",
    "The notification system seems completely broken on my end.",
    "I keep getting kicked out of the app randomly.",
    "The interface glitches and buttons don't respond when I tap them.",
]

TECHNICAL_DETAILS = [
    "I'm using version {version} on {device}.",
    "This started happening after the last update.",
    "I've tried restarting my {device} but the problem persists.",
    "It works fine on my laptop but not on my {device}.",
    "I've already cleared the cache and reinstalled the app.",
    "The error code I'm seeing is ERR-{error_code}.",
    "This has been going on for {days} days now.",
    "I've tried using different browsers but same issue.",
    "My internet connection is fine — other apps work normally.",
    "Other users in my household are experiencing the same thing.",
    "I attached a screenshot but can't include it here.",
    "I've already tried all the basic troubleshooting steps.",
    "",
    "",
    "",
]

TECHNICAL_CLOSINGS = [
    "Can someone please help me resolve this?",
    "I need this fixed as soon as possible.",
    "Is there a workaround I can try in the meantime?",
    "Please escalate this if necessary.",
    "This is really affecting my work.",
    "I rely on this app daily and can't afford downtime.",
    "How can I get this working again?",
    "Is this a known issue? Are you working on a fix?",
    "I'd appreciate any help with this problem.",
    "Can you provide a timeline for when this will be resolved?",
    "I'm losing patience with this recurring issue.",
    "I've spent hours trying to fix this myself.",
    "",
    "",
]

# ──────────────────────────────────────────────
# ACCOUNT templates
# ──────────────────────────────────────────────
ACCOUNT_PASSWORD_OPENINGS = [
    "I forgot my password and the reset link isn't working.",
    "I need to change my password but I can't find the option.",
    "The password reset email never arrives in my inbox.",
    "I tried resetting my password but I get an error.",
    "I'm locked out because I forgot my password.",
    "My password expired and I can't seem to create a new one.",
    "I've been trying to reset my password for the past hour.",
    "The reset link I received says it has expired.",
    "I can't remember my password and need to regain access.",
    "How do I change my password? The current one doesn't work.",
    "I haven't been able to update my password through your website.",
    "I keep getting 'invalid token' when I click the reset link.",
    "My temporary password stopped working before I could set a new one.",
    "I requested a password reset but didn't get any email.",
    "Can someone manually reset my password? The automated system isn't working for me.",
]

ACCOUNT_LOGIN_OPENINGS = [
    "I can't log into my account even though I'm using the correct credentials.",
    "Every time I try to sign in, it says 'invalid credentials'.",
    "I'm locked out of my account after too many failed attempts.",
    "The login page just keeps refreshing without letting me in.",
    "I enter my email and password but nothing happens.",
    "I get an error saying my account doesn't exist, but I've had it for years.",
    "Two-factor authentication isn't sending me the verification code.",
    "My account seems to be suspended but I don't know why.",
    "I haven't been able to access my account since last week.",
    "The login button is greyed out and I can't click it.",
    "I'm getting a 'session expired' message immediately after logging in.",
    "I can't sign in from my new phone.",
    "My account shows as deactivated but I never deactivated it.",
    "Single sign-on with Google isn't working for my account anymore.",
    "The CAPTCHA on the login page keeps failing even when I solve it correctly.",
]

ACCOUNT_DETAILS = [
    "My registered email is {email}.",
    "My username is {username}.",
    "I've been a customer since {year}.",
    "I've verified my email address already.",
    "I can provide ID verification if needed.",
    "I've already tried from multiple devices.",
    "This is urgent as I need access for work.",
    "I checked my spam folder but nothing is there.",
    "",
    "",
    "",
]

ACCOUNT_CLOSINGS = [
    "Please help me regain access.",
    "I need this sorted out as soon as possible.",
    "What steps do I need to take to fix this?",
    "Can you send me a manual reset link?",
    "Is there another way to verify my identity?",
    "I'd appreciate a quick resolution.",
    "This is becoming very frustrating.",
    "I can't access any of my saved data without logging in.",
    "Please advise on what to do next.",
    "",
    "",
]

# ──────────────────────────────────────────────
# SHIPPING templates
# ──────────────────────────────────────────────
SHIPPING_OPENINGS = [
    "My order still hasn't arrived and it's been over two weeks.",
    "I need to track my package but the tracking number doesn't work.",
    "Where is my order? The estimated delivery date has passed.",
    "The tracking information hasn't been updated in five days.",
    "I placed an order on {date} and it still hasn't been delivered.",
    "My package seems to be stuck in transit.",
    "I received a delivery confirmation but the package isn't here.",
    "Can you tell me where my order is right now?",
    "The courier says they delivered it but I never received anything.",
    "My order has been 'out for delivery' for three days straight.",
    "I want to know why my package is taking so long.",
    "The delivery was supposed to arrive by {date}.",
    "I'm worried my package got lost in transit.",
    "Tracking shows my order is in a completely different city.",
    "My parcel has been sitting at the distribution center for a week.",
    "I haven't received any shipping updates since I placed my order.",
    "The estimated arrival keeps getting pushed back.",
    "Is there a way to expedite my delivery?",
    "I paid for express delivery but it's been {days} days already.",
    "My neighbor says no delivery attempt was made at my address.",
]

SHIPPING_DETAILS = [
    "My order number is {order_id}.",
    "The tracking number is {tracking_id}.",
    "I ordered on {date}.",
    "I chose standard delivery.",
    "I selected express delivery.",
    "My delivery address is correct — I've double checked.",
    "I've contacted the courier but they redirected me to you.",
    "This is a gift and I needed it by a specific date.",
    "",
    "",
    "",
]

SHIPPING_CLOSINGS = [
    "Can you check the status of my delivery?",
    "I need to know when I'll receive my order.",
    "Please provide an updated delivery estimate.",
    "If it's lost, I need a replacement or refund.",
    "I'd like to file a complaint about the delivery service.",
    "Can you contact the courier on my behalf?",
    "This delay is really inconvenient.",
    "I need someone to investigate what happened to my package.",
    "Is there any way to speed this up?",
    "I'm very disappointed with the delivery experience.",
    "",
    "",
]

# ──────────────────────────────────────────────
# REFUND templates
# ──────────────────────────────────────────────
REFUND_OPENINGS = [
    "I was charged twice and I want my money back for the extra payment.",
    "I received the wrong item and want to return it for a full reimbursement.",
    "The product I received is defective and I want my money back.",
    "I returned my order two weeks ago but still haven't gotten my money back.",
    "I'm not satisfied with my purchase and would like to return it.",
    "I canceled my order but the charge still went through.",
    "The item doesn't match the description and I want my money returned.",
    "I was overcharged and need the difference returned to my card.",
    "My order arrived damaged and I need a full reimbursement.",
    "I never received my order but I was charged for it.",
    "The duplicate charge on my card needs to be reversed immediately.",
    "The service didn't work as advertised — I want my money back.",
    "I've been waiting over a month for the money to come back to my account.",
    "I was told I would get my payment back but it hasn't appeared on my statement.",
    "I was charged twice by mistake — can I get the extra amount returned?",
    "I paid twice by mistake and need one payment returned.",
    "The quality of what I received doesn't justify the price.",
    "The product broke within the first day — I need my money back.",
    "I accidentally purchased the wrong plan and need my money back.",
    "My return was approved weeks ago but the credit still hasn't appeared.",
]

REFUND_DETAILS = [
    "The amount I need returned is ${amount}.",
    "I was charged ${amount} twice.",
    "The order number is {order_id}.",
    "I returned the item via {carrier} on {date}.",
    "The return tracking shows it was delivered back to your warehouse.",
    "I have the receipt and can provide proof of the duplicate charge.",
    "I already spoke with someone who said it would be taken care of.",
    "It's been {days} days since I requested my money back.",
    "",
    "",
    "",
]

REFUND_CLOSINGS = [
    "Please return my payment as soon as possible.",
    "How long will it take to get my money back?",
    "I need the amount credited to my original payment method.",
    "Can you expedite this reimbursement?",
    "I want a full reimbursement, not store credit.",
    "Please confirm when the reversal has been issued.",
    "This has gone on too long — I want my money back now.",
    "I'm going to dispute the charge with my bank if this isn't resolved.",
    "I just want my money returned.",
    "Is there anything else you need from me to process this?",
    "",
    "",
]

# ──────────────────────────────────────────────
# SUBSCRIPTION templates
# ──────────────────────────────────────────────
SUBSCRIPTION_OPENINGS = [
    "I want to cancel my monthly plan.",
    "How do I stop my recurring payment?",
    "I'd like to end my membership effective immediately.",
    "Please cancel my auto-renewal — I don't want to be charged again.",
    "I no longer need the service and want to cancel.",
    "I've decided to discontinue my plan.",
    "Can you help me cancel before the next payment cycle?",
    "I tried to cancel online but the button doesn't seem to work.",
    "I want to unsubscribe from the premium tier.",
    "I need to downgrade or cancel my current plan.",
    "I signed up by mistake and need to cancel right away.",
    "The service isn't worth the monthly fee anymore.",
    "I want to stop all future charges related to my plan.",
    "Please terminate my membership and confirm via email.",
    "I'm switching to a different provider and need to cancel here.",
    "I've been trying to cancel for days but can't figure out how.",
    "I don't want to renew my annual plan.",
    "Can I cancel mid-cycle and get a prorated refund?",
    "I want to make sure I won't be charged next month.",
    "Please remove my payment method and cancel the recurring charge.",
]

SUBSCRIPTION_DETAILS = [
    "My plan is the {plan} tier at ${amount}/month.",
    "I've been a member since {year}.",
    "My next payment date is {date}.",
    "I'm currently on the {plan} plan.",
    "I already tried to cancel through the website.",
    "I don't see a cancel option in my account settings.",
    "I read the cancellation policy but still have questions.",
    "",
    "",
    "",
]

SUBSCRIPTION_CLOSINGS = [
    "Please confirm the cancellation in writing.",
    "I need a confirmation email once it's done.",
    "Make sure no further charges are applied.",
    "Can you also delete my stored payment information?",
    "I don't want any retention offers — just cancel please.",
    "How soon will the cancellation take effect?",
    "Will I still have access until the end of the current period?",
    "Thank you for processing this promptly.",
    "Just want this taken care of.",
    "",
    "",
]


# ══════════════════════════════════════════════
# TONE / STYLE MODIFIERS
# ══════════════════════════════════════════════

FRUSTRATED_PREFIXES = [
    "I am extremely frustrated. ",
    "This is absolutely unacceptable. ",
    "I've had it with your service! ",
    "Honestly, this is the worst experience. ",
    "I can't believe this is happening again. ",
    "This is ridiculous. ",
    "I'm really upset about this. ",
    "Seriously?? ",
    "I am so done with this. ",
    "NOT HAPPY AT ALL. ",
    "I've wasted so much time on this already. ",
    "This is the third time I'm reaching out about the same issue. ",
]

POLITE_PREFIXES = [
    "Hello, I hope you're having a good day. ",
    "Hi there, I have a question. ",
    "Good morning, I'd like some help please. ",
    "Dear support team, ",
    "Hi, I hope this message finds you well. ",
    "Hello, thank you in advance for your help. ",
    "Good afternoon, I was wondering if you could assist me. ",
    "Hi, sorry to bother you but ",
    "Greetings, I'm writing to inquire about something. ",
]

POLITE_SUFFIXES = [
    " Thank you for your time.",
    " Thanks in advance!",
    " I appreciate your help.",
    " Many thanks.",
    " Looking forward to hearing from you.",
    " Thanks so much.",
    " I really appreciate it.",
    " Regards.",
    " Best wishes.",
]

# ──────────────────────────────────────────────
# Random fill-in values
# ──────────────────────────────────────────────
AMOUNTS = [
    "9.99", "14.99", "19.99", "24.99", "29.99", "39.99",
    "49.99", "59.99", "79.99", "99.99", "149.99", "199.99",
]

DEVICES = [
    "iPhone", "Android phone", "iPad", "Windows laptop",
    "MacBook", "Samsung tablet", "desktop computer",
]

PLANS = ["Basic", "Standard", "Premium", "Pro", "Enterprise", "Starter"]

CARRIERS = ["UPS", "FedEx", "USPS", "DHL", "the courier"]

DATES = [
    "October 1st", "September 28th", "last Monday", "two days ago",
    "last week", "three days ago", "yesterday", "last Friday",
    "on the 15th", "earlier this week", "a few days ago",
]

VERSIONS = ["3.2.1", "4.0.0", "2.8.5", "5.1.0", "3.9.2", "4.2.3"]

YEARS = ["2021", "2022", "2023", "2024"]

USERNAMES = [
    "jsmith42", "alex.jones", "maria_garcia", "customer2024",
    "user12345", "david.lee", "sarah_w", "techuser99",
]

EMAILS = [
    "j***@email.com", "a***@gmail.com", "m***@yahoo.com",
    "d***@outlook.com", "user***@mail.com",
]

# ──────────────────────────────────────────────
# Common typo replacements (applied randomly)
# ──────────────────────────────────────────────
TYPO_MAP = {
    "received": "recieved",
    "definitely": "definately",
    "their": "thier",
    "because": "becuase",
    "occurred": "occured",
    "separate": "seperate",
    "tomorrow": "tommorow",
    "immediately": "immediatly",
    "unfortunately": "unfortunatly",
    "convenience": "convienence",
    "inconvenience": "inconvienence",
    "experience": "experiance",
    "subscription": "subsription",
    "canceled": "cancled",
    "payment": "payement",
    "address": "adress",
    "access": "acess",
    "appreciate": "apreciate",
    "doesn't": "doesnt",
    "haven't": "havent",
    "wasn't": "wasnt",
    "couldn't": "couldnt",
}


# ══════════════════════════════════════════════
# PRIORITY & SENTIMENT DISTRIBUTIONS PER CATEGORY
# ══════════════════════════════════════════════
# These probability weights control how priority and sentiment
# are distributed per category, making the dataset realistic.
# (e.g., billing issues skew higher priority and negative sentiment)

CATEGORY_PRIORITY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "billing":      {"low": 0.05, "medium": 0.25, "high": 0.50, "urgent": 0.20},
    "technical":    {"low": 0.10, "medium": 0.35, "high": 0.40, "urgent": 0.15},
    "account":      {"low": 0.10, "medium": 0.40, "high": 0.35, "urgent": 0.15},
    "shipping":     {"low": 0.15, "medium": 0.45, "high": 0.30, "urgent": 0.10},
    "refund":       {"low": 0.05, "medium": 0.20, "high": 0.50, "urgent": 0.25},
    "subscription": {"low": 0.20, "medium": 0.45, "high": 0.25, "urgent": 0.10},
}

CATEGORY_SENTIMENT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "billing":      {"positive": 0.05, "neutral": 0.25, "negative": 0.70},
    "technical":    {"positive": 0.05, "neutral": 0.30, "negative": 0.65},
    "account":      {"positive": 0.05, "neutral": 0.45, "negative": 0.50},
    "shipping":     {"positive": 0.05, "neutral": 0.35, "negative": 0.60},
    "refund":       {"positive": 0.05, "neutral": 0.15, "negative": 0.80},
    "subscription": {"positive": 0.10, "neutral": 0.50, "negative": 0.40},
}


# ══════════════════════════════════════════════
# TEMPLATE POOLS (category → (openings, details, closings))
# ══════════════════════════════════════════════

TEMPLATE_POOLS: Dict[str, Tuple[List[str], List[str], List[str]]] = {
    "billing":      (BILLING_OPENINGS,          BILLING_DETAILS,      BILLING_CLOSINGS),
    "technical":    (TECHNICAL_OPENINGS,         TECHNICAL_DETAILS,    TECHNICAL_CLOSINGS),
    "shipping":     (SHIPPING_OPENINGS,          SHIPPING_DETAILS,     SHIPPING_CLOSINGS),
    "refund":       (REFUND_OPENINGS,            REFUND_DETAILS,       REFUND_CLOSINGS),
    "subscription": (SUBSCRIPTION_OPENINGS,      SUBSCRIPTION_DETAILS, SUBSCRIPTION_CLOSINGS),
}

# Account has two sub-types (password reset vs login issues)
ACCOUNT_POOLS = {
    "reset_password":     (ACCOUNT_PASSWORD_OPENINGS, ACCOUNT_DETAILS, ACCOUNT_CLOSINGS),
    "resolve_login_issue": (ACCOUNT_LOGIN_OPENINGS,   ACCOUNT_DETAILS, ACCOUNT_CLOSINGS),
}


# ══════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════

def _fill_placeholders(text: str, rng: random.Random) -> str:
    """Replace placeholders like {amount}, {date}, etc. with random values."""
    replacements = {
        "{amount}":     rng.choice(AMOUNTS),
        "${amount}":    "$" + rng.choice(AMOUNTS),
        "{date}":       rng.choice(DATES),
        "{order_id}":   f"ORD-{rng.randint(100000, 999999)}",
        "{tracking_id}": f"TRK{rng.randint(1000000000, 9999999999)}",
        "{device}":     rng.choice(DEVICES),
        "{version}":    rng.choice(VERSIONS),
        "{error_code}": str(rng.randint(1000, 9999)),
        "{days}":       str(rng.randint(2, 14)),
        "{plan}":       rng.choice(PLANS),
        "{year}":       rng.choice(YEARS),
        "{email}":      rng.choice(EMAILS),
        "{username}":   rng.choice(USERNAMES),
        "{carrier}":    rng.choice(CARRIERS),
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    return text


def _maybe_add_typos(text: str, rng: random.Random, prob: float = 0.15) -> str:
    """With probability `prob`, introduce one minor spelling mistake."""
    if rng.random() > prob:
        return text
    # Pick one word to replace with its typo version
    candidates = [w for w in TYPO_MAP if w in text.lower()]
    if not candidates:
        return text
    word = rng.choice(candidates)
    # Replace only the first occurrence (case-insensitive)
    idx = text.lower().find(word)
    if idx == -1:
        return text
    original_word = text[idx:idx + len(word)]
    typo = TYPO_MAP[word]
    # Preserve capitalization of first char
    if original_word[0].isupper():
        typo = typo[0].upper() + typo[1:]
    return text[:idx] + typo + text[idx + len(word):]


def _maybe_modify_punctuation(text: str, rng: random.Random) -> str:
    """Randomly modify punctuation for variety."""
    roll = rng.random()
    if roll < 0.05:
        # Remove trailing period
        text = text.rstrip(".")
    elif roll < 0.10:
        # Add extra exclamation marks
        if text.endswith("."):
            text = text[:-1] + "!!"
        elif text.endswith("!"):
            text = text + "!"
    elif roll < 0.13:
        # Add ellipsis
        if text.endswith("."):
            text = text[:-1] + "..."
    return text


def _apply_tone(
    text: str,
    sentiment: str,
    rng: random.Random,
) -> str:
    """Optionally prepend/append tone modifiers based on sentiment."""
    if sentiment == "negative" and rng.random() < 0.35:
        text = rng.choice(FRUSTRATED_PREFIXES) + text
    elif sentiment == "positive" and rng.random() < 0.40:
        text = rng.choice(POLITE_PREFIXES) + text
        if rng.random() < 0.5:
            text = text.rstrip(".!") + rng.choice(POLITE_SUFFIXES)
    elif sentiment == "neutral" and rng.random() < 0.20:
        text = rng.choice(POLITE_PREFIXES) + text
    return text


def _generate_ticket_text(
    category: str,
    resolution: str,
    sentiment: str,
    rng: random.Random,
) -> str:
    """
    Build one customer-support ticket text by combining fragments
    from the appropriate pools.
    """
    # Select the right template pools
    if category == "account":
        pools = ACCOUNT_POOLS[resolution]
    else:
        pools = TEMPLATE_POOLS[category]

    openings, details, closings = pools

    # Pick one piece from each pool
    opening = rng.choice(openings)
    detail = rng.choice(details)
    closing = rng.choice(closings)

    # Fill placeholders
    opening = _fill_placeholders(opening, rng)
    detail = _fill_placeholders(detail, rng)
    closing = _fill_placeholders(closing, rng)

    # Assemble — sometimes skip detail or closing for variation
    parts = [opening]
    if detail and rng.random() < 0.75:
        parts.append(detail)
    if closing and rng.random() < 0.70:
        parts.append(closing)

    text = " ".join(p for p in parts if p)

    # Apply tone
    text = _apply_tone(text, sentiment, rng)

    # Random modifications for realism
    text = _maybe_add_typos(text, rng, prob=0.15)
    text = _maybe_modify_punctuation(text, rng)

    # Occasionally make the whole message lowercase (casual style)
    if rng.random() < 0.08:
        text = text.lower()

    # Occasionally make part of it ALL CAPS (frustrated customer)
    if sentiment == "negative" and rng.random() < 0.08:
        words = text.split()
        if len(words) > 3:
            start = rng.randint(0, min(3, len(words) - 2))
            end = min(start + rng.randint(2, 5), len(words))
            words[start:end] = [w.upper() for w in words[start:end]]
            text = " ".join(words)

    return text.strip()


def _weighted_choice(
    options: List[str],
    weights: Dict[str, float],
    rng: random.Random,
) -> str:
    """Pick from `options` using probability `weights`."""
    w = [weights[opt] for opt in options]
    return rng.choices(options, weights=w, k=1)[0]


# ══════════════════════════════════════════════
# MAIN GENERATION FUNCTION
# ══════════════════════════════════════════════

def generate_dataset(
    num_samples: int = NUM_SAMPLES,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic customer-support ticket dataset.

    Parameters
    ----------
    num_samples : int
        Number of tickets to generate (default 5000).
    seed : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ticket_id, text, category, priority,
        sentiment, language, resolution.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    logger.info(f"Generating {num_samples} synthetic tickets with seed={seed}")

    # Distribute samples roughly evenly across categories
    samples_per_category = num_samples // len(CATEGORIES)
    remainder = num_samples % len(CATEGORIES)

    category_counts = {}
    for i, cat in enumerate(CATEGORIES):
        category_counts[cat] = samples_per_category + (1 if i < remainder else 0)

    logger.info(f"Target distribution: {category_counts}")

    records = []
    seen_texts = set()  # Track unique texts to avoid duplicates
    ticket_num = 0

    for category, count in category_counts.items():
        # Get valid resolutions for this category
        valid_resolutions = CATEGORY_RESOLUTION_MAP[category]

        for _ in range(count):
            ticket_num += 1
            ticket_id = f"TKT-{ticket_num:05d}"

            # Pick resolution (uniform among valid ones for this category)
            resolution = rng.choice(valid_resolutions)

            # Pick priority (weighted by category)
            priority = _weighted_choice(
                PRIORITIES, CATEGORY_PRIORITY_WEIGHTS[category], rng
            )

            # Pick sentiment (weighted by category)
            sentiment = _weighted_choice(
                SENTIMENTS, CATEGORY_SENTIMENT_WEIGHTS[category], rng
            )

            # Generate ticket text — retry if duplicate
            max_retries = 50
            text = None
            for attempt in range(max_retries):
                candidate = _generate_ticket_text(
                    category, resolution, sentiment, rng
                )
                if candidate not in seen_texts:
                    text = candidate
                    seen_texts.add(text)
                    break
            else:
                # After max retries, force uniqueness with a suffix
                text = candidate + f" [ref:{ticket_id}]"
                if text in seen_texts:
                    text = candidate + f" (case {ticket_num})"
                seen_texts.add(text)
                logger.warning(
                    f"Duplicate retry exhausted for {ticket_id}, "
                    f"appended unique suffix."
                )

            records.append({
                "ticket_id":  ticket_id,
                "text":       text,
                "category":   category,
                "priority":   priority,
                "sentiment":  sentiment,
                "language":   "english",
                "resolution": resolution,
            })

    # Shuffle so tickets aren't grouped by category
    rng.shuffle(records)

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} tickets.")
    return df


# ══════════════════════════════════════════════
# VALIDATION FUNCTION
# ══════════════════════════════════════════════

def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Validate the generated dataset for correctness.

    Checks:
        1. Number of records
        2. Column names
        3. Missing values
        4. Duplicate ticket IDs
        5. Duplicate ticket text
        6-10. Invalid labels (category, priority, sentiment, language, resolution)

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to validate.

    Returns
    -------
    bool
        True if all checks pass, False otherwise.
    """
    all_ok = True
    logger.info("=" * 60)
    logger.info("DATASET VALIDATION")
    logger.info("=" * 60)

    # 1. Number of records
    n = len(df)
    if n == NUM_SAMPLES:
        logger.info(f"[PASS] Record count: {n} (expected {NUM_SAMPLES})")
    else:
        logger.error(f"[FAIL] Record count: {n} (expected {NUM_SAMPLES})")
        all_ok = False

    # 2. Column names
    expected_cols = set(REQUIRED_COLUMNS)
    actual_cols = set(df.columns.tolist())
    if expected_cols == actual_cols:
        logger.info(f"[PASS] Columns: {sorted(actual_cols)}")
    else:
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols
        logger.error(f"[FAIL] Missing columns: {missing}, Extra columns: {extra}")
        all_ok = False

    # 3. Missing values
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    if total_missing == 0:
        logger.info("[PASS] No missing values")
    else:
        logger.error(f"[FAIL] Missing values found:\n{missing_counts[missing_counts > 0]}")
        all_ok = False

    # 4. Duplicate ticket IDs
    dup_ids = df["ticket_id"].duplicated().sum()
    if dup_ids == 0:
        logger.info("[PASS] No duplicate ticket IDs")
    else:
        logger.error(f"[FAIL] Duplicate ticket IDs: {dup_ids}")
        all_ok = False

    # 5. Duplicate ticket text
    dup_texts = df["text"].duplicated().sum()
    if dup_texts == 0:
        logger.info("[PASS] No duplicate ticket text")
    else:
        logger.error(f"[FAIL] Duplicate ticket texts: {dup_texts}")
        all_ok = False

    # 6-10. Validate label values
    for field, valid_values in VALID_LABELS.items():
        invalid = df[~df[field].isin(valid_values)]
        if len(invalid) == 0:
            logger.info(f"[PASS] All '{field}' values are valid")
        else:
            bad_vals = invalid[field].unique().tolist()
            logger.error(
                f"[FAIL] Invalid '{field}' values found: {bad_vals} "
                f"({len(invalid)} rows)"
            )
            all_ok = False

    # Additional: check category-resolution consistency
    inconsistent = 0
    for _, row in df.iterrows():
        valid_res = CATEGORY_RESOLUTION_MAP.get(row["category"], [])
        if row["resolution"] not in valid_res:
            inconsistent += 1
    if inconsistent == 0:
        logger.info("[PASS] All category-resolution combinations are valid")
    else:
        logger.error(
            f"[FAIL] {inconsistent} rows have impossible "
            f"category-resolution combinations"
        )
        all_ok = False

    logger.info("=" * 60)
    if all_ok:
        logger.info("VALIDATION RESULT: ALL CHECKS PASSED ✓")
    else:
        logger.error("VALIDATION RESULT: SOME CHECKS FAILED ✗")
    logger.info("=" * 60)

    return all_ok


# ══════════════════════════════════════════════
# SAVE FUNCTION
# ══════════════════════════════════════════════

def save_dataset(df: pd.DataFrame, path: Path = RAW_TICKETS_PATH) -> None:
    """Save the dataset to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Dataset saved to: {path}")
    logger.info(f"File size: {path.stat().st_size:,} bytes")


# ══════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════

def main() -> None:
    """Generate, validate, and save the synthetic dataset."""
    logger.info("=" * 60)
    logger.info("TicketTriage-SLM — Synthetic Data Generation")
    logger.info("=" * 60)

    # 1. Generate
    df = generate_dataset()

    # 2. Validate
    is_valid = validate_dataset(df)
    if not is_valid:
        logger.error("Dataset validation failed. Aborting save.")
        raise ValueError("Generated dataset failed validation checks.")

    # 3. Save
    save_dataset(df)

    # 4. Print summary statistics
    logger.info("")
    logger.info("=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)

    print(f"\nShape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print(f"\n5 randomly selected tickets (seed=123 for display):")
    print(df.sample(5, random_state=123).to_string(index=False))

    print(f"\nCategory distribution:")
    print(df["category"].value_counts().to_string())

    print(f"\nPriority distribution:")
    print(df["priority"].value_counts().to_string())

    print(f"\nSentiment distribution:")
    print(df["sentiment"].value_counts().to_string())

    print(f"\nResolution distribution:")
    print(df["resolution"].value_counts().to_string())

    print(f"\nMissing values per column:")
    print(df.isnull().sum().to_string())

    print(f"\nDuplicate ticket IDs: {df['ticket_id'].duplicated().sum()}")
    print(f"Duplicate ticket texts: {df['text'].duplicated().sum()}")

    logger.info("=" * 60)
    logger.info("Phase 2 complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

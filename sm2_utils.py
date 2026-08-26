from datetime import datetime, timedelta
from typing import Dict, Tuple

def calculate_sm2(quality: int, repetition_count: int, ease_factor: float, interval_days: int) -> Tuple[int, float, int, str]:
    """
    Calculates updated SM-2 parameters based on recall performance quality (0 to 3).
    Returns: (new_repetition_count, new_ease_factor, new_interval_days, next_review_iso_str)
    """
    # Clamp quality grade between 0 and 3
    q = max(0, min(3, quality))

    # 1. Update Ease Factor (EF)
    new_ef = ease_factor + (0.1 - (3 - q) * (0.08 + (3 - q) * 0.02))
    if new_ef < 1.3:
        new_ef = 1.3

    # 2. Update Repetition Count & Interval
    if q < 2:
        # Failed recall -> Reset progress
        new_rep = 0
        new_interval = 1
    else:
        # Successful recall -> Scale progress
        new_rep = repetition_count + 1
        if new_rep == 1:
            new_interval = 1
        elif new_rep == 2:
            new_interval = 6
        else:
            new_interval = int(round(interval_days * new_ef))

    # 3. Calculate Next Review Timestamp (ISO format)
    next_review_dt = datetime.now() + timedelta(days=new_interval)
    next_review_str = next_review_dt.isoformat()

    return new_rep, round(new_ef, 2), new_interval, next_review_str
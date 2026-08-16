"""Verbatim prompt texts for the teacher check. Committed before the first forward pass.

All three prompts emit the A/B answer **first**. Any checklist or justification is
requested only for text that follows the answer token, so it cannot condition the answer.
Because the score is read from the probability over the two option tokens at the first
answer position, the three prompts differ **only in the instruction framing that precedes
the answer** — the generation path is identical. That is deliberate: this round tests
whether the model can hear the attribute, not whether it can describe it.

Option order is counterbalanced. Each recording is scored under both orders and the two
probabilities are averaged, so a positional preference cannot look like hearing.

No prompt may mention COVID, a diagnosis, or the words dry/wet outside the option list.
"""

OPTIONS = ("dry", "wet")

_HEAD = ("You are listening to a single cough recording. "
         "Judge only what is audible in the recording.")

_ASK = ("Which of the two options describes the cough?\n"
        "A) {a}\nB) {b}\n"
        "Respond with exactly one letter, A or B, as the very first character.")

PROMPTS = {
    "plain":
        _HEAD + "\n\n" + _ASK + "\n\nAnswer:",

    "cue":
        _HEAD + "\n\n" + _ASK + "\n"
        "After that single letter, list on separate lines what you heard for each of: "
        "burst sharpness, presence of liquid or bubbling, voicing, duration, "
        "background noise.\n\nAnswer:",

    "cue_evidence":
        _HEAD + "\n\n" + _ASK + "\n"
        "After that single letter, list on separate lines what you heard for each of: "
        "burst sharpness, presence of liquid or bubbling, voicing, duration, "
        "background noise — and for each, add a short clause naming the moment in the "
        "recording it came from.\n\nAnswer:",
}


def render(prompt_id, swap=False):
    """swap=True presents the options in the reversed order; the caller must invert the
    mapping from letter to class when reading the probabilities back."""
    a, b = (OPTIONS[1], OPTIONS[0]) if swap else OPTIONS
    return PROMPTS[prompt_id].format(a=a, b=b)


def positive_letter(swap):
    """Which letter means 'wet', the positive class, under this option order."""
    return "A" if swap else "B"


if __name__ == "__main__":
    for pid in PROMPTS:
        for sw in (False, True):
            print(f"===== {pid}  swap={sw}  positive={positive_letter(sw)} =====")
            print(render(pid, sw))
            print()

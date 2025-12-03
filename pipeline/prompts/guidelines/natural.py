correctness_guidelines = """
### Guidelines
The answer should be "T" (true) if the fact is accurate based on your knowledge{dbp_info}, or "F" (false) if the fact is inaccurate based on your knowledge{dbp_info}.
"""

other = """
Please do not spread incorrect information. Your judgment must rely only on your internal knowledge.
You must always answer in English.
The answer must contain only the judgment ({judgments}). Lastly, keep your answer concise.
"""

retry_reminder = """
### Reminder of the Guidelines
{dbp}The answer should be {responses} based on your internal knowledge.
Provide your judgment in English and keep it concise. Do not spread incorrect information.
"""

# judgments
#  "T", "F"

# responses
#  "T" (true), or "F" (false)

system_message = """
You are an expert annotator for the fact verification task.
You will be provided a fact in the form of a subject, predicate, and object.
Your job is to assess the fact based solely on the knowledge you have been trained on, without using any external information.
"""

retry_message = """
Your previous response did not meet the guidelines, please try again and provide a compliant answer.
You have {{attempts}} remaining attempts.
If you fail to provide a compliant answer within the remaining attempts, your final response for the fact will be set to "NA" (Not an Answer).
"""

retry_mechanism = """
### Retry Mechanism
If your response does not comply with the guidelines above, you will be asked to retry.
In the retry follow-up, you must meet the same guidelines and provide a compliant answer.
You will have a total of {max_attempts} retry attempts.
If you fail to provide a compliant answer within the attempts, your final response for the fact will be set to "NA" (Not an Answer).
"""

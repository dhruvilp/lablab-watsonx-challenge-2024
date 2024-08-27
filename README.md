# lablab-watsonx-challenge-2024

[![Watch the demo](https://img.youtube.com/vi/RvJ0WMWJE0g/0.jpg)](https://youtu.be/RvJ0WMWJE0g)

DEMO: https://youtu.be/RvJ0WMWJE0g

Financial institutions in the financial services sector face challenges in complying with evolving AI regulations that mandate transparency, accountability, and data privacy. They are required to implement robust governance frameworks to document AI decision-making processes, conduct regular audits, and ensure the ethical use of AI in high-stakes areas like AML and BSA compliance. Institutions must also adapt to changing regulatory requirements, provide clear explanations of AI model decisions, safeguard against biases, and maintain data privacy. https://thefinancialbrand.com/news/banking-trends-strategies/genai-bankings-best-offense-in-challenging-times-174940/ The proposed solution addresses the compliance challenges by automating key aspects of regulatory monitoring. By using Whisper to transcribe relevant communications, Granite LLM Model to analyze data, and MoA to delegate and automate tasks, the solution ensures efficient tracking of regulatory changes and internal policies. Langchain manages the workflow to maintain timely compliance, while Crew AI coordinates between agents and human officers, enhancing overall adherence to regulations and streamlining the compliance process.

## Possible use cases

Use Case: A pharmaceutical company wants to ensure compliance with Good Manufacturing Practices (GMP).
Required Data Sources:
* Regulatory Data: FDA guidance documents, GMP regulations, and industry standards.
* Internal Policy Data: Company policies on quality control, documentation, and training.
* Employee Activity Data: Records of employee training, production records, and quality control inspections.
* External Data: Industry news articles on GMP updates, regulatory enforcement actions, and quality control best practices.

Use Case: A financial services company wants to ensure compliance with anti-money laundering (AML) regulations.
Required Data Sources:
* Regulatory Data: Regulations from the Financial Crimes Enforcement Network (FinCEN), Securities and Exchange Commission (SEC), and other relevant authorities.
* Internal Policy Data: Company policies on customer due diligence, transaction monitoring, and suspicious activity reporting.
* Employee Activity Data: Records of employee training, customer interactions, and transaction reviews.
* External Data: News articles on AML trends, regulatory enforcement actions, and industry best practices.

Use Case: A healthcare provider wants to ensure compliance with HIPAA regulations.
Required Data Sources:
* Regulatory Data: HIPAA Privacy Rule, Security Rule, and Breach Notification Rule.
* Internal Policy Data: Company policies on data security, access controls, and incident response.
* Employee Activity Data: Records of employee training, access logs, and incident reports.
* External Data: News articles on HIPAA breaches, enforcement actions, and cybersecurity best practices.

Use Case: A manufacturing company wants to ensure compliance with environmental regulations.
Required Data Sources:
* Regulatory Data: Environmental Protection Agency (EPA) regulations, state and local environmental laws.
* Internal Policy Data: Company policies on waste management, emissions control, and environmental impact assessments.
* Employee Activity Data: Records of employee training, environmental monitoring data, and incident reports.
* External Data: News articles on environmental regulations, enforcement actions, and industry best practices.

## Proposed Solution
![image](https://raw.githubusercontent.com/dhruvilp/lablab-watsonx-challenge-2024/main/watsonx-streamlit-demo.png)

The proposed solution addresses the compliance challenges by automating key aspects of regulatory monitoring. 
By using Granite LLM Model to analyze data, and MoA to delegate and automate tasks, the solution ensures efficient tracking of regulatory changes and internal policies. 
Langchain manages the workflow to maintain timely compliance, while Crew AI coordinates between agents and human officers, enhancing overall adherence to regulations and streamlining the compliance process.

## Tech Stack
* Crew AI
* IBM WatsonX AI
* IBM Granite 13B V2 Chat LLM
* Meta Llama 2 Chat LLM

## Sample Prompts

```
Prompt 1: Regulatory Update and Policy Alignment
User Prompt: "Please review the latest FinCEN alerts and advisories to ensure we are up-to-date with current AML regulations. Based on the updates, adjust our internal policies accordingly and prepare a compliance report detailing any changes or areas of concern."
Expected Tasks Triggered: Regulatory Change Tracking, Policy Review and Update, Compliance Report Generation
-------
Prompt 2: Suspicious Activity Monitoring
User Prompt: "Monitor all recent transactions for any suspicious activity that could indicate potential money laundering. Generate alerts if suspicious activity is detected and compile a report to be sent to the relevant regulatory bodies."
Expected Tasks Triggered: Suspicious Activity Monitoring and Reporting, Regulatory Communication and Liaison
-------
Prompt 3: Internal Audit and Risk Assessment
User Prompt: "Conduct an internal audit to assess our current AML compliance. Identify any gaps or weaknesses in our processes and evaluate our AML risk exposure. Based on your findings, develop strategies to mitigate any identified risks."
Expected Tasks Triggered: Internal Audit and Compliance Assessment, Risk Assessment and Mitigation
-------
Prompt 4: Employee Training and Onboarding
User Prompt: "Ensure all new employees complete their AML training and background checks before they start. Also, review and refresh our AML training materials to ensure they are up-to-date with the latest policies. Schedule training sessions for all current employees to cover any updates."
Expected Tasks Triggered: Employee Screening and Onboarding Compliance, AML Compliance Training Refresh
-------
Prompt 5: Regulatory Communication and Compliance Updates
User Prompt: "Maintain communication with AML regulatory bodies to stay informed about any new requirements or advisories. Ensure that these updates are communicated to the relevant internal stakeholders and that necessary adjustments are made to our compliance practices."
Expected Tasks Triggered: Regulatory Communication and Liaison, Policy Review and Update (if needed)
-------
Prompt 6: Detailed Compliance Report
User Prompt: "Generate a comprehensive compliance report that includes details on recent audit findings, any updates to internal policies, and a summary of employee training status. Ensure the report is updated on our compliance dashboards."
Expected Tasks Triggered: Compliance Report Generation
-------
Prompt 7: Training Coordination for Updated Compliance Practices
User Prompt: "Coordinate and track the scheduling of employee training sessions focused on the latest AML compliance practices. Ensure all employees are notified and that their training status is recorded for future audits."
Expected Tasks Triggered: Employee Training Coordination, AML Compliance Training Refresh (if updating materials)
-------
Prompt 8: Policy Review and External Compliance Liaison
User Prompt: "Review our internal policies for any necessary updates based on recent FinCEN advisories. Communicate any policy changes to the relevant regulatory bodies and ensure all adjustments are implemented."
Expected Tasks Triggered: Policy Review and Update, Regulatory Communication and Liaison
```
### Set up the environment

-  Create a <a href="https://cloud.ibm.com/catalog/services/watson-machine-learning" target="_blank" rel="noopener no referrer">Watson Machine Learning (WML) Service</a> instance (a free plan is offered and information about how to create the instance can be found <a href="https://dataplatform.cloud.ibm.com/docs/content/wsj/admin/create-services.html?context=wx&audience=wdp" target="_blank" rel="noopener no referrer">here</a>).

### Defining the WML credentials
This cell defines the WML credentials required to work with watsonx Foundation Model inferencing.
**Action:** Provide the IBM Cloud user API key. For details, see <a href="https://cloud.ibm.com/iam/apikeys" target="_blank" rel="noopener no referrer">documentation</a>.

### How-to Run

#### Developer Notes
```sh

python3 -m venv env
source env/bin/activate
deactivate

pip install -r requirements.txt

streamlit run fincomply-streamlit.py 

```

#### Author
Dhruvil Patel, MBA, Full-Stack Software Engineer and AI Enthusiast.

import os
from langchain_core.callbacks import BaseCallbackHandler
from typing import TYPE_CHECKING, Any, Dict, Optional
from crewai_tools import tool
from crewai import Agent, Task, Crew, Process
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_ibm import WatsonxLLM
from langchain_ibm import ChatWatsonx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

IBM_CLOUD_URL = "https://us-south.ml.cloud.ibm.com"
WML_KEY = os.getenv("WML_KEY")
PROJECT_ID = "c2e1f1d0-2f15-495b-94a9-ed84335489e6"

avators = {
    "agent": "https://cdn-icons-png.flaticon.com/512/15861/15861185.png",
    "human": "https://cdn-icons-png.flaticon.com/512/17593/17593656.png"
}

class MyCustomHandler(BaseCallbackHandler):

    
    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        st.session_state.messages.append({"role": "assistant", "content": inputs['input']})
        st.chat_message(self.agent_name, avatar=avators[self.agent_name]).write(f"---agent-start: {inputs['input']}")
   
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        st.session_state.messages.append({"role": self.agent_name, "content": outputs['output']})
        st.chat_message(self.agent_name, avatar=avators[self.agent_name]).write(f"---agent-end: {outputs['output']}")
        

################################################################################################
#                                        IBM
################################################################################################        

project_id = PROJECT_ID

parameters = {
    GenParams.MIN_NEW_TOKENS: 0,
    GenParams.MAX_NEW_TOKENS: 2049, # 1024, 2049, 3073, 4096, 
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.REPETITION_PENALTY: 1
}

granite_model = WatsonxLLM(
    model_id="ibm/granite-13b-chat-v2",
    url=IBM_CLOUD_URL,
    project_id=project_id,
    apikey=WML_KEY,
    params=parameters
)

llama_model = WatsonxLLM(
    model_id="meta-llama/llama-2-70b-chat", # "meta-llama/llama-2-70b-chat"
    url=IBM_CLOUD_URL,
    project_id=project_id,
    apikey=WML_KEY,
    params=parameters
)

granite_chat = ChatWatsonx(
    model_id="ibm/granite-13b-chat-v2",
    url=IBM_CLOUD_URL,
    project_id=project_id,
    apikey=WML_KEY,
    params=parameters,
)

llama_chat = ChatWatsonx(
    model_id="meta-llama/llama-2-70b-chat",
    url=IBM_CLOUD_URL,
    project_id=project_id,
    apikey=WML_KEY,
    params=parameters,
)

################################################################################################
#                                        TOOLS
################################################################################################
# FinCen Source of Data: https://www.fincen.gov/resources/advisoriesbulletinsfact-sheets

@tool
def get_fincen_alerts() -> str:
    """Gets up-to-date information on FinCen alerts.
    """
    # In real life, these data can be retrieved from appropriate source via API call
    mocked_alerts = [
        {
            "title": "FIN-2024-Alert002",
            "date": "2024-08-25",
            "description": "FinCEN Issues Supplemental Alert on Israeli Extremist Violence in the West Bank",
            "alert_message": "Supplemental Alert: FinCEN Highlights Additional Red Flags Regarding Financing of Israeli Extremist Settler Violence Against Palestinians in the West Bank As noted in the February 2024 Alert, the United States has consistently opposed actions that undermine peace, security, or stability in the West Bank, including attacks by Israeli violent extremist settlers against Palestinians, attacks by Palestinian violent extremists against Israelis, and other acts that unduly restrict civilians’ access to essential services and basic necessities and risk further escalating tensions and expanding the conflict in the region.3 Treasury continues to be concerned by reports of escalating violence in the West Bank,4 including reported attacks by Israeli settlers on humanitarian aid convoys,5 and encourages continued reporting by financial institutions of suspicious activity potentially related to the financing of these violent acts. Since publication of the February 2024 Alert, Treasury’s Office of Foreign Assets Control (OFAC) and the Department of State have collectively designated 13 individuals and entities6 under Executive Order (E.O.) 14115, which authorizes the U.S. government to impose financial sanctions on foreign persons that, inter alia, are responsible for or complicit in, or have directly or indirectly engaged or attempted to engage in (1) actions that threaten the peace, security, or stability of the West Bank; or (2) planning, ordering, otherwise directing, or participating in specified actions affecting the West Bank, such as violence targeting civilians and property destruction.7 OFAC reminds individuals and financial institutions that all U.S. persons, which generally include any person in the United States, any entity organized under U.S. law, and any U.S. citizen or legal permanent resident anywhere in the world, are generally prohibited from making any contribution or provision of funds, goods, or services by, to, or for the benefit of any individual or entity designated under E.O. 14115 and any entities owned 50 percent or more by such designated individuals or entities.8 This includes engaging in crowdfunding activities that are used to raise funds internationally in support of designated violent extremist Israeli settlers, as well as other fundraising undertaken on behalf of such designated persons.9 Non-U.S. persons are also subject to certain OFAC prohibitions, such as prohibitions on causing or conspiring to cause U.S. persons to violate U.S. sanctions, as well as engaging in conduct that evades U.S. sanctions. Violations of U.S. sanctions may result in civil or criminal penalties,10 and persons who materially support any person blocked pursuant to E.O. 14115 may themselves be sanctioned. FinCEN is providing additional red flag indicators of suspicious activity potentially related to the financing of Israeli extremist settler violence against Palestinians in the West Bank, including flags derived from open-source reporting. All red flag indicators highlighted in the February 2024 Alert remain valid,11 and FinCEN has identified the following additional red flag indicators to help financial institutions detect, prevent, and report potential suspicious activity related to the financing of Israeli extremist settler violence against Palestinians in the West Bank: Payments involving entities, individuals, addresses on accounts, receiving addresses, or IP addresses linked to any West Bank “outpost.”12 References or payments to accounts held in the name of Israeli violent extremist settler groups or individuals involved or implicated in violence against Palestinians in the West Bank, including references or payments to accounts held by known immediate family members of designated individuals. The West Bank region may also be referred to as “Judea and Samaria,” “Shomron,” “Yehuda,” or “Yehuda v’Shomoron” by supporters of Israeli settlements in the West Bank.13 References or payments to accounts or webpages that are fundraising for convicted, imprisoned, or U.S.-sanctioned violent extremists who targeted Palestinians in the West Bank. Accounts that share a merchant representative, email, phone number, social media page, or physical address with accounts held in the name of designated individuals or entities. Payments to beneficiaries, accounts, or crowdfunding campaigns linked to social media indicating support for violence in the West Bank, including pictures and videos, support for violence targeting Palestinians in the West Bank, or references to known or designated Israeli settler violent extremists, outposts, herding farms, or groups. Crowdfunding campaigns and social media may include phrases like “State Land,” “Area C,” “preserving State Land,” “Hilltop Youth,” “Youth of the Hills,” or “stopping supply convoys.” FinCEN reminds financial institutions that, as no single red flag is necessarily indicative of illicit or suspicious activity, financial institutions should consider all the surrounding facts and circumstances before determining whether a specific transaction is suspicious or associated with potential Israeli extremist settler violence."
        },
        {
            "title": "FIN-2024-Alert001",
            "date": "2024-02-01",
            "description": "FinCEN Issues Alert on Israeli Extremist Settler Violence Against Palestinians in the West Bank",
            "alert_message": "FinCEN Alert on Israeli Extremist Settler Violence Against Palestinians in the West Bank The Financial Crimes Enforcement Network (FinCEN) is issuing this alert to financial institutions1 related to the financing of Israeli extremist settler violence against Palestinians in the West Bank. This alert provides select red flags to assist financial institutions in identifying and reporting suspicious activity that finances such violence. The United States has consistently opposed actions that undermine peace, security, or stability in the West Bank, including attacks by Israeli violent extremist settlers against Palestinians, attacks by Palestinian violent extremists against Israelis, and other acts that unduly restrict civilians’ access to essential services and basic necessities and risk further escalating tensions and expanding the conflict in the region.2 As the U.S. Department of State (State) noted, the U.S. Government “will continue to seek accountability and justice for all acts of violence against civilians in the West Bank, regardless of the perpetrator or the victim.”3 Treasury’s Office of Foreign Assets Control (OFAC) additionally is highlighting for members of the public that the President, on February 1, 2024, issued an Executive Order (E.O.) that authorizes the U.S. government to impose sanctions on foreign persons that are responsible for or complicit in, or have directly or indirectly engaged or attempted to engage in (1) actions that threaten the peace, security, or stability of the West Bank; or (2) planning, ordering, otherwise directing, or participating in specified actions affecting the West Bank, such as violence targeting civilians and property destruction.4 The United States seeks to impose tangible and significant consequences on those engaged in such activities, as well as to protect the U.S. financial system from abuse FinCEN has identified the following red flag indicators to help detect, prevent, and report potential suspicious activity related to the financing of Israeli extremist settler violence against Palestinians in the West Bank. As no single red flag is necessarily indicative of illicit or suspicious activity, financial institutions should consider all the surrounding facts and circumstances before determining whether a specific transaction is suspicious or associated with potential Israeli extremist settler violence: Payments to any organizations or groups, including nonprofit organizations (NPO),5 that are now or have been previously linked to violent extremist groups in the West Bank or lists an officer, founder, or director that is currently or has been previously linked to Israeli violent extremist groups in the West Bank. Information included in a transaction between customers, such as references in the memo field, that indicate support for Israeli violent extremist groups or campaigns. Transactions with no apparent economic, business, or lawful purpose associated with a rapid movement of funds and linked to NPOs active in supporting violent extremist Israeli settlers in the West Bank, particularly if the NPO has advocated for, or solicited donations on social media in support of, Israeli violent extremist groups or campaigns. Purchases of tactical military gear for resale overseas and destined for non-government Israeli end-users in the West Bank, particularly if the end-users are currently or have been previously linked to Israeli violent extremist groups in the West Bank. FinCEN requests that financial institutions reference this alert by including the key term “FIN-2024- WBEXTREMISM” in SAR field 2 (Filing Institutions Note to FinCEN) and the narrative to indicate a connection between the suspicious activity being reported and this alert."
        },
        {
            "title": "FIN-2024-Alert003",
            "date": "2024-08-15",
            "description": "FinCEN Alert on Emerging Trends in Human Trafficking Financing",
            "alert_message": "FinCEN has issued an alert to financial institutions regarding new trends in the financing of human trafficking operations. The alert outlines emerging methods used by traffickers to conceal illicit funds, including the use of shell companies, cryptocurrency transactions, and complex layering schemes. Financial institutions are advised to enhance their monitoring systems to detect transactions linked to known human trafficking networks and to report suspicious activities that may be related to the financing of trafficking operations. Key red flags include unusual transactions involving high-value goods or services, payments to entities associated with human trafficking, and the use of anonymizing technologies in financial transactions."
        },
        {
            "title": "FIN-2024-Alert004",
            "date": "2024-07-01",
            "description": "FinCEN Alert on Financial Support to Non-State Armed Groups in Conflict Zones",
            "alert_message": "FinCEN is alerting financial institutions about the increasing risk of financial support to non-state armed groups operating in conflict zones. The alert highlights recent patterns of financial transactions that support groups involved in armed conflict, including large transfers to entities in conflict zones, and donations to organizations with known links to armed groups. Financial institutions are encouraged to apply enhanced due diligence measures for transactions related to high-risk regions and to report any suspicious activity that may indicate financial support to such groups. Specific red flags include transactions involving politically exposed persons (PEPs) in conflict regions, payments to entities with ties to armed conflict, and use of front companies to obscure the true nature of transactions."
        },
        {
            "title": "FIN-2024-Alert005",
            "date": "2024-08-12",
            "description": "FinCEN Alert on Sanctions Evasion Through Trade-Based Money Laundering",
            "alert_message": "FinCEN has issued a new alert to address the risks of sanctions evasion through trade-based money laundering (TBML). The alert outlines various TBML techniques used by sanctioned entities to bypass financial sanctions, including over- and under-invoicing, false documentation, and the use of intermediaries to obscure transaction details. Financial institutions are advised to scrutinize trade transactions involving high-risk jurisdictions and industries known for TBML practices. Key red flags include discrepancies between invoiced and declared values of goods, complex shipping routes with multiple intermediaries, and transactions involving entities with a history of sanctions violations. Enhanced due diligence is recommended for transactions that show signs of trade-based anomalies or suspicious patterns."
        },
        {
            "title": "FIN-2024-Alert006",
            "date": "2024-07-23",
            "description": "FinCEN Alert on Increased Threat of Financial Cyberattacks Targeting Financial Institutions",
            "alert_message": "FinCEN has issued an alert to financial institutions regarding the rising threat of financial cyberattacks. The alert provides information on recent cyber threats and attacks targeting financial institutions, including ransomware attacks, phishing schemes, and advanced persistent threats (APTs). Financial institutions are advised to implement robust cybersecurity measures, including regular updates to security protocols, employee training on recognizing phishing attempts, and enhanced monitoring for unusual network activity. Red flags include unusual login patterns, unexpected requests for sensitive information, and the presence of malicious software on institutional systems. Financial institutions are encouraged to report any suspicious cyber activities promptly to help mitigate the risks associated with financial cyberattacks."
        }
    ]
    return mocked_alerts


@tool
def get_fincen_advisories() -> str:
    """Gets recent information on FinCen Advisories.
    """
    # In real life, these data can be retrieved from appropriate source via API call
    mocked_advisories = [
        {
            "title": "FinCEN Advisory FIN-2024-A001",
            "date": "2024-05-08",
            "description": "Advisory to Counter the Financing of Iran-Backed Terrorist Organizations",
            "alert_message": "FinCEN Advisory to Financial Institutions to Counter the Financing of Iran-Backed Terrorist Organizations Introduction The Financial Crimes Enforcement Network (FinCEN) is issuing this advisory to assist financial institutions in detecting potentially illicit transactions related to Islamic Republic of Iran (Iran)-backed terrorist organizations amid intensified terrorist activity in the Middle East.1 Recent events have underscored Iran’s involvement in and financing of terrorist activity in the region. Iran seeks, among other goals, to project power by exporting terrorism throughout the Middle East and beyond through the financing of a range of regional armed groups, some of which are U.S.-designated Foreign Terrorist Organizations (FTOs) or Specially Designated Global Terrorist organizations (SDGTs).2 These terrorist organizations include Lebanese Hizballah (Hizballah),3 Hamas, the Palestinian Islamic Jihad (PIJ),5 the Houthis (Ansarallah),6 and several Iran-aligned militia groups in Iraq and Syria. As demonstrated by the October 7, 2023, Hamas attack on Israel and recent Houthi attacks in the Red Sea,7 these organizations are capable of perpetrating horrific violence, causing destruction, and disrupting critical supply chains. The U.S. Department of the Treasury (Treasury) has been systematically working to dismantle these organizations by disrupting their illicit finance networks and eliminating their sources of revenue.8 This advisory highlights the means by which terrorist organizations receive support from Iran and describes several typologies these terrorist organizations use to illicitly access or circumvent the international financial system to raise, move, and spend funds. It also provides red flags that may assist financial institutions in identifying related suspicious activity and is consistent with FinCEN’s National Anti-Money Laundering and Countering the Financing of Terrorism (AML/ CFT) Priorities, which include terrorist financing.9 The information contained in this advisory is derived from FinCEN’s analysis of Bank Secrecy Act (BSA) data, open-source reporting, and information provided by law enforcement partners. Iran’s Sources of Foreign Revenue Iran uses the revenue from the sale of commodities, particularly oil, to countries such as the People’s Republic of China (PRC) to fund its terrorist proxies.13 Following the reimposition of U.S. sanctions against Iran’s petroleum sector in 2018,14 Iran’s ability to finance itself through sales of crude oil and petroleum products—its most important economic sector—was severely diminished. In response, Iran established large-scale global oil smuggling and money laundering networks to enable access to foreign currency and the international financial system through the illicit sale of crude oil and petroleum products in global markets.15 In 2021, the National Iranian Oil Company sold approximately $40 billion worth of products,16 and its crude oil and condensate exports reached an average of more than 600,000 barrels per day, with nearly all of it sent to the PRC and Syria.17 Iran’s exports to the PRC have increased over time, reaching approximately 1.3 million barrels per day in 2023.18 Some of these oil proceeds finance the activities of the IRGC-QF and other terrorist groups.19 To restrict these sources of revenue, Treasury has designated numerous Iranian- and third-country operatives, front companies, and ships involved in Iran’s oil smuggling networks."
        },
        {
            "title": "OFAC Advisory OFAC-2024-A002",
            "date": "2024-06-15",
            "description": "Advisory on Sanctions Evasion Tactics in the Maritime Sector",
            "alert_message": "The Office of Foreign Assets Control (OFAC) has issued this advisory to inform financial institutions and maritime companies about emerging tactics used by sanctioned entities to evade sanctions through maritime operations. The advisory details common evasion strategies, including the use of shell companies, falsified documentation, and deceptive ship-to-ship transfers. It provides guidance on identifying suspicious maritime transactions and recommends enhanced due diligence procedures to prevent facilitating illicit activities linked to sanctioned entities."
        },
        {
            "title": "FATF Advisory FATF-2024-A003",
            "date": "2024-07-25",
            "description": "FATF Recommendations for Combating Money Laundering in the Real Estate Sector",
            "alert_message": "The Financial Action Task Force (FATF) has released updated recommendations aimed at strengthening anti-money laundering (AML) measures in the real estate sector. This advisory outlines best practices for real estate transactions, including enhanced customer due diligence, suspicious activity reporting, and the monitoring of high-value transactions. It emphasizes the need for real estate professionals to be vigilant against money laundering schemes and to adhere to AML regulations to ensure the integrity of real estate markets."
        },
        {
            "title": "SEC Advisory SEC-2024-A004",
            "date": "2024-08-10",
            "description": "SEC Guidance on Compliance with New Investment Fund Regulations",
            "alert_message": "The Securities and Exchange Commission (SEC) has issued guidance regarding new regulations affecting investment funds. This advisory outlines the requirements for compliance with updated rules on disclosure, reporting, and risk management for investment funds. It provides detailed instructions on implementing the new regulatory framework and emphasizes the importance of transparency and accuracy in fund reporting. Financial institutions are advised to review their practices and ensure alignment with the latest SEC regulations."
        },
        {
            "title": "Basel Committee Advisory BCBS-2024-A001",
            "date": "2024-08-20",
            "description": "Basel Committee Recommendations on Operational Risk Management",
            "alert_message": "The Basel Committee on Banking Supervision (BCBS) has issued recommendations for improving operational risk management within banking institutions. This advisory highlights best practices for identifying, assessing, and mitigating operational risks, including those related to technology, fraud, and business continuity. It outlines the requirements for robust risk management frameworks and emphasizes the need for regular risk assessments and internal controls to safeguard against operational disruptions."
        }
    ]
    return mocked_advisories
   
    
@tool
def get_fincen_notices() -> str:
    """Gets up-to-date information on FinCen Notices.
    """
    # In real life, these data can be retrieved from appropriate source via API call
    mocked_notices = [
        {
            "title": "FinCEN Notice FIN-2024-NTC2",
            "date": "2024-07-16",
            "description": "FinCEN, OFAC, and FBI Joint Notice on Timeshare Fraud Associated with Mexico-Based Transnational Criminal Organizations",
            "alert_message": "FinCEN Advisory to Financial Institutions to Counter the Financing of Iran-Backed Terrorist Organizations Introduction The Financial Crimes Enforcement Network (FinCEN) is issuing this advisory to assist financial institutions in detecting potentially illicit transactions related to Islamic Republic of Iran (Iran)-backed terrorist organizations amid intensified terrorist activity in the Middle East.1 Recent events have underscored Iran’s involvement in and financing of terrorist activity in the region. Iran seeks, among other goals, to project power by exporting terrorism throughout the Middle East and beyond through the financing of a range of regional armed groups, some of which are U.S.-designated Foreign Terrorist Organizations (FTOs) or Specially Designated Global Terrorist organizations (SDGTs).2 These terrorist organizations include Lebanese Hizballah (Hizballah),3 Hamas, the Palestinian Islamic Jihad (PIJ),5 the Houthis (Ansarallah),6 and several Iran-aligned militia groups in Iraq and Syria. As demonstrated by the October 7, 2023, Hamas attack on Israel and recent Houthi attacks in the Red Sea,7 these organizations are capable of perpetrating horrific violence, causing destruction, and disrupting critical supply chains. The U.S. Department of the Treasury (Treasury) has been systematically working to dismantle these organizations by disrupting their illicit finance networks and eliminating their sources of revenue.8 This advisory highlights the means by which terrorist organizations receive support from Iran and describes several typologies these terrorist organizations use to illicitly access or circumvent the international financial system to raise, move, and spend funds. It also provides red flags that may assist financial institutions in identifying related suspicious activity and is consistent with FinCEN’s National Anti-Money Laundering and Countering the Financing of Terrorism (AML/ CFT) Priorities, which include terrorist financing.9 The information contained in this advisory is derived from FinCEN’s analysis of Bank Secrecy Act (BSA) data, open-source reporting, and information provided by law enforcement partners. Iran’s Sources of Foreign Revenue Iran uses the revenue from the sale of commodities, particularly oil, to countries such as the People’s Republic of China (PRC) to fund its terrorist proxies.13 Following the reimposition of U.S. sanctions against Iran’s petroleum sector in 2018,14 Iran’s ability to finance itself through sales of crude oil and petroleum products—its most important economic sector—was severely diminished. In response, Iran established large-scale global oil smuggling and money laundering networks to enable access to foreign currency and the international financial system through the illicit sale of crude oil and petroleum products in global markets.15 In 2021, the National Iranian Oil Company sold approximately $40 billion worth of products,16 and its crude oil and condensate exports reached an average of more than 600,000 barrels per day, with nearly all of it sent to the PRC and Syria.17 Iran’s exports to the PRC have increased over time, reaching approximately 1.3 million barrels per day in 2023.18 Some of these oil proceeds finance the activities of the IRGC-QF and other terrorist groups.19 To restrict these sources of revenue, Treasury has designated numerous Iranian- and third-country operatives, front companies, and ships involved in Iran’s oil smuggling networks."
        },
        {
            "title": "OFAC Notice OFAC-2024-NTC1",
            "date": "2024-06-10",
            "description": "OFAC Advisory on Sanctions Compliance Related to Cryptocurrency Transactions",
            "alert_message": "The Office of Foreign Assets Control (OFAC) is issuing this advisory to alert financial institutions about the risks associated with cryptocurrency transactions involving sanctioned entities. The advisory provides guidelines for compliance with OFAC regulations when dealing with digital assets and addresses common typologies of evasion attempts by sanctioned parties. Financial institutions are advised to implement enhanced due diligence procedures for transactions involving cryptocurrency and to ensure that their compliance programs adequately address the risks posed by digital assets."
        },
        {
            "title": "FATF Notice FATF-2024-NTC3",
            "date": "2024-08-22",
            "description": "FATF Guidance on Anti-Money Laundering Measures for Virtual Assets and Virtual Asset Service Providers",
            "alert_message": "The Financial Action Task Force (FATF) has released updated guidance for Anti-Money Laundering (AML) measures applicable to virtual assets and virtual asset service providers (VASPs). The guidance includes recommendations for risk assessment, customer due diligence, and transaction monitoring specific to virtual assets. It emphasizes the need for jurisdictions to enforce AML regulations on VASPs and improve international cooperation to address the risks associated with virtual asset transactions."
        },
        {
            "title": "SEC Notice SEC-2024-NTC5",
            "date": "2024-07-30",
            "description": "SEC Alert on Enhanced Reporting Requirements for Large Financial Institutions",
            "alert_message": "The Securities and Exchange Commission (SEC) has issued a notice regarding enhanced reporting requirements for large financial institutions. This notice outlines new regulations for the disclosure of financial risks, including stress testing and liquidity management. The SEC emphasizes the need for institutions to adopt robust risk management frameworks and to provide comprehensive reports on their financial stability and exposure to systemic risks. Compliance with these enhanced requirements is critical for maintaining market integrity."
        },
        {
            "title": "Basel Committee Notice BCBS-2024-NTC2",
            "date": "2024-08-05",
            "description": "Basel Committee Guidance on Risk Management Frameworks for Banking Institutions",
            "alert_message": "The Basel Committee on Banking Supervision (BCBS) has released updated guidance on risk management frameworks for banking institutions. This notice details best practices for identifying, assessing, and mitigating risks across various banking operations. It includes recommendations for enhancing internal controls, improving risk governance structures, and implementing effective risk mitigation strategies. Financial institutions are encouraged to review and update their risk management frameworks in line with the new guidance to ensure regulatory compliance and financial stability."
        }
    ]
    return mocked_notices


@tool
def get_internal_policy_docs(department: str) -> str:
    """Gets internal policy documents based on current date
    
    Params:
    department: regulatory department type (Ex: compliance, AML, etc)
    """
    # In real life, these data can be retrieved from appropriate source via API call
    mocked_policies = [
        {
            "policy_id": "INT-PAY-001",
            "policy_title": "Cross-Border Payments Compliance",
            "description": "This policy outlines the comprehensive guidelines and procedures that must be followed to ensure full compliance with all relevant regulations governing cross-border payments. It includes detailed steps for documentation, mandatory reporting requirements, and the specific roles and responsibilities of various departments in managing cross-border transactions. The policy also provides best practices for mitigating risks associated with international payment processing and includes regular audits to ensure ongoing adherence to compliance standards.",
            "effective_date": "2022-01-15",
            "last_reviewed_date": "2023-01-15",
            "department": "financial crimes",
            "compliance_requirement": "High"
        },
        {
            "policy_id": "INT-PAY-002",
            "policy_title": "Anti-Money Laundering (AML) for International Transfers",
            "description": "This policy provides a thorough framework for conducting Anti-Money Laundering (AML) checks on international transfers. It includes detailed procedures for customer due diligence, transaction monitoring, and the identification and reporting of suspicious activities in line with global AML regulations. The policy also covers the training requirements for staff involved in international transfers, ensuring they are equipped with the knowledge and tools necessary to detect and prevent money laundering activities. Additionally, the policy mandates regular reviews and updates to stay aligned with evolving AML legislation.",
            "effective_date": "2022-03-01",
            "last_reviewed_date": "2023-02-28",
            "department": "AML Department",
            "compliance_requirement": "High"
        },
        {
            "policy_id": "INT-PAY-003",
            "policy_title": "Regulatory Reporting Requirements",
            "description": "This policy outlines the requirements for regulatory reporting to ensure compliance with financial regulations. It includes guidelines for the accurate and timely submission of reports to regulatory bodies, documentation standards, and the responsibilities of different departments in the reporting process. The policy also covers the procedures for managing changes in reporting requirements and ensures adherence to both local and international regulations.",
            "effective_date": "2023-03-01",
            "last_reviewed_date": "2024-03-01",
            "department": "Compliance",
            "compliance_requirement": "High"
        },
        {
            "policy_id": "INT-PAY-004",
            "policy_title": "Global AML Compliance Framework",
            "description": "This policy provides a comprehensive framework for Anti-Money Laundering (AML) compliance in line with global standards. It includes procedures for customer due diligence, transaction monitoring, suspicious activity reporting, and the implementation of AML controls across international operations. The policy ensures that all AML practices are consistent with international regulations and incorporates regular updates to reflect changes in global AML requirements.",
            "effective_date": "2022-06-01",
            "last_reviewed_date": "2023-06-01",
            "department": "AML Department",
            "compliance_requirement": "Critical"
        },
        {
            "policy_id": "INT-PAY-005",
            "policy_title": "Securities Trading Compliance",
            "description": "This policy sets forth the standards for compliance with regulations governing securities trading. It includes guidelines for insider trading prevention, trading practices, reporting obligations, and the management of conflicts of interest. The policy ensures that securities trading activities adhere to regulatory requirements and international standards, with regular monitoring and audits to maintain compliance.",
            "effective_date": "2023-02-01",
            "last_reviewed_date": "2024-02-01",
            "department": "Compliance",
            "compliance_requirement": "High"
        },
        {
            "policy_id": "INT-PAY-006",
            "policy_title": "Foreign Exchange Transactions Regulation",
            "description": "This policy details the regulatory requirements for foreign exchange (FX) transactions. It includes procedures for transaction reporting, risk management, and compliance with international FX regulations. The policy ensures that all FX transactions are conducted in accordance with regulatory standards and includes measures for monitoring and reporting irregularities or breaches.",
            "effective_date": "2022-11-01",
            "last_reviewed_date": "2023-11-01",
            "department": "Treasury",
            "compliance_requirement": "High"
        },
        {
            "policy_id": "INT-PAY-007",
            "policy_title": "Privacy Standards for Financial Data",
            "description": "This policy establishes the privacy standards for handling and protecting financial data. It includes guidelines for data collection, storage, and sharing, ensuring compliance with global data privacy regulations such as GDPR and CCPA. The policy outlines the roles and responsibilities of staff in maintaining data privacy and mandates regular audits to verify compliance with privacy standards.",
            "effective_date": "2023-07-01",
            "last_reviewed_date": "2024-07-01",
            "department": "Compliance",
            "compliance_requirement": "High"
        }
    ]
    
    filtered_data = [
        item for item in mocked_policies 
        if item['department'].lower() == department.lower()
    ]
    
    return filtered_data if filtered_data else [{"message": f"No policy documents found for department: {department}"}]

  
    
@tool
def get_audit_findings() -> str:
    """Gets internal audit findings based on current date
    """
    # In real life, these data can be retrieved from appropriate source via API call
    mocked_audit_findings = [
        {
            "audit_id": "AUD-2023-001",
            "audit_title": "Sanctions Screening Effectiveness Audit",
            "findings": "The audit of sanctions screening processes found that while most transactions were appropriately screened, there were gaps in the screening of certain types of payments, particularly those involving correspondent banks. The audit also identified issues with the frequency of sanctions list updates, which were not always implemented promptly. Recommended actions include automating the update process and expanding the scope of screening procedures.",
            "audit_date": "2024-08-15",
            "department_audited": ["Compliance", "AML Department", "Sanctions and Screening Department"],
            "risk_level": "High",
            "recommended_actions": "Automate sanctions list updates, expand the scope of screening to include all payment types, and establish a more frequent review cycle for screening processes."
        },
        {
            "audit_id": "AUD-2024-002",
            "audit_title": "Data Privacy Compliance Review",
            "findings": "The audit of data privacy practices revealed that while most data handling procedures complied with regulations, there were significant gaps in the management of third-party data processors. The audit found that data protection agreements were not consistently reviewed or updated, and there were delays in addressing data subject access requests. Recommended actions include establishing a formal process for reviewing third-party data agreements, improving the tracking of data access requests, and enhancing staff training on data privacy regulations.",
            "audit_date": "2024-05-10",
            "department_audited": ["IT Security", "Data Privacy Department", "Legal Department"],
            "risk_level": "Medium",
            "recommended_actions": "Implement a regular review process for third-party data agreements, enhance tracking and response procedures for data subject access requests, and provide additional training on data privacy regulations."
        },
        {
            "audit_id": "AUD-2024-003",
            "audit_title": "Financial Reporting Accuracy Check",
            "findings": "The audit of financial reporting accuracy found that while the overall financial statements were correct, there were discrepancies in the reporting of revenue recognition and expense allocation. Issues were identified with the consistency of applying accounting principles and documentation practices. Recommended actions include revising the revenue recognition policy, improving the documentation of expense allocation decisions, and conducting periodic internal reviews to ensure adherence to accounting standards.",
            "audit_date": "2024-06-22",
            "department_audited": ["Finance", "Accounting Department", "Internal Audit"],
            "risk_level": "High",
            "recommended_actions": "Update the revenue recognition policy, enhance documentation practices for expense allocation, and establish periodic internal audits to ensure compliance with accounting standards."
        },
        {
            "audit_id": "AUD-2024-004",
            "audit_title": "IT System Security Assessment",
            "findings": "The audit of IT system security highlighted several vulnerabilities in the network infrastructure and access controls. It was found that outdated software patches and weak password policies were contributing to potential security risks. The audit also identified a lack of regular security training for employees. Recommended actions include implementing a robust patch management process, strengthening password policies, and scheduling regular security awareness training for all staff members.",
            "audit_date": "2024-07-18",
            "department_audited": ["IT Department", "Network Security", "HR Department"],
            "risk_level": "Critical",
            "recommended_actions": "Establish a comprehensive patch management process, enforce stronger password policies, and implement regular security training programs for employees."
        }
    ]
    return mocked_audit_findings
  
    
@tool
def get_compliance_training_materials(department: str) -> str:
    """Gets compliance training materials for a given department
    
    Params:
    department: regulatory department type (Ex: compliance, AML, etc)
    """
    
    mocked_training_materials = [
        {
            "department": "AML Department",
            "training_materials": [
                {
                    "title": "Anti-Money Laundering (AML) Compliance",
                    "duration": "3 hours",
                    "format": "Online Module",
                    "summary": "This module provides an in-depth understanding of anti-money laundering laws and regulations. It teaches employees how to identify suspicious activities, report them, and ensure compliance with AML requirements."
                },
                {
                    "title": "Financial Reporting Standards",
                    "duration": "2 hours",
                    "format": "Webinar",
                    "summary": "A webinar covering the latest financial reporting standards, including IFRS and GAAP. It is essential for employees to stay updated on reporting requirements to ensure accuracy and compliance."
                }
            ]
        },
        {
            "department": "Compliance",
            "training_materials": [
                {
                    "title": "Ethical Sales Practices",
                    "duration": "2 hours",
                    "format": "Video",
                    "summary": "A video training that outlines ethical sales practices, including honesty in advertising, customer transparency, and avoiding conflicts of interest. It ensures that sales teams adhere to ethical standards."
                },
                {
                    "title": "Customer Data Protection",
                    "duration": "1.5 hours",
                    "format": "Online Module",
                    "summary": "This module teaches sales teams how to protect customer data, emphasizing the importance of confidentiality, secure data handling, and compliance with data protection laws."
                },
                {
                    "title": "Data Privacy and Protection",
                    "duration": "1 hour",
                    "format": "Interactive Course",
                    "summary": "An essential course for understanding data privacy laws such as GDPR. It explains how to handle personal data responsibly and ensure compliance with data protection regulations."
                }
            ]
        },
        {
            "department": "Financial Crimes",
            "training_materials": [
                {
                    "title": "Ethical Sales Practices",
                    "duration": "2 hours",
                    "format": "Video",
                    "summary": "A video training that outlines ethical sales practices, including honesty in advertising, customer transparency, and avoiding conflicts of interest. It ensures that sales teams adhere to ethical standards."
                },
                {
                    "title": "Customer Data Protection",
                    "duration": "1.5 hours",
                    "format": "Online Module",
                    "summary": "This module teaches sales teams how to protect customer data, emphasizing the importance of confidentiality, secure data handling, and compliance with data protection laws."
                },
                {
                    "title": "Data Privacy and Protection",
                    "duration": "1 hour",
                    "format": "Interactive Course",
                    "summary": "An essential course for understanding data privacy laws such as GDPR. It explains how to handle personal data responsibly and ensure compliance with data protection regulations."
                }
            ]
        }
    ]

    filtered_data = [
        item for item in mocked_training_materials 
        if item['department'].lower() == department.lower()
    ]
    
    return filtered_data if filtered_data else [{"message": f"No training materials found for department: {department}"}]


@tool
def get_risk_assessment_data(risk_category: str) -> str:
    """Gets compliance risk assessment data for a given risk category
    
    Params:
    risk_category: compliance risk category (Ex: compliance, legal, etc)
    """
    
    mock_risk_assessment_data = [
        {
            "risk_id": "001",
            "risk_category": "financial",
            "risk_description": "Potential non-compliance with anti-money laundering (AML) regulations due to insufficient transaction monitoring.",
            "risk_level": "high",
            "mitigation_measures": [
                "Enhance transaction monitoring systems.",
                "Implement stricter due diligence procedures.",
                "Regularly audit transaction reports."
            ],
            "assessment_date": "2024-08-01"
        },
        {
            "risk_id": "002",
            "risk_category": "legal",
            "risk_description": "Risk of violating international sanctions due to unverified beneficiary details in cross-border payments.",
            "risk_level": "medium",
            "mitigation_measures": [
                "Verify beneficiary details against international sanctions lists.",
                "Implement real-time transaction screening.",
                "Provide training on compliance with international sanctions."
            ],
            "assessment_date": "2024-08-15"
        },
        {
            "risk_id": "003",
            "risk_category": "operational",
            "risk_description": "Operational risk due to potential errors in international payment processing systems.",
            "risk_level": "medium",
            "mitigation_measures": [
                "Conduct regular system audits.",
                "Improve accuracy of data entry procedures.",
                "Implement fail-safes and error-checking protocols."
            ],
            "assessment_date": "2024-08-22"
        },
        {
            "risk_id": "004",
            "risk_category": "compliance",
            "risk_description": "Risk of non-compliance with GDPR requirements for handling customer data in cross-border transactions.",
            "risk_level": "high",
            "mitigation_measures": [
                "Ensure data protection policies align with GDPR.",
                "Conduct regular data privacy audits.",
                "Implement robust data encryption measures."
            ],
            "assessment_date": "2024-08-10"
        },
        {
            "risk_id": "005",
            "risk_category": "financial",
            "risk_description": "Potential for financial losses due to exchange rate fluctuations in international payments.",
            "risk_level": "low",
            "mitigation_measures": [
                "Use hedging strategies to manage currency risk.",
                "Monitor exchange rates and adjust payment schedules accordingly.",
                "Provide currency risk management training for finance teams."
            ],
            "assessment_date": "2024-08-18"
        }
    ]

    
    filtered_data = [
        item for item in mock_risk_assessment_data 
        if item['risk_category'].lower() == risk_category.lower()
    ]
    
    return filtered_data if filtered_data else [{"message": f"No risk assessment data found for risk category: {risk_category}"}]


@tool
def get_legal_opinions(regulation: str) -> str:
    """Gets internal legal experts opinions related to specific regulations or compliance issues for a given regulation type
    
    Params:
    regulation: regulation type (Ex: international sanctions, AML, etc)
    """
    
    mock_legal_opinions = [
        {
            "opinion_id": "101",
            "regulation": "gdpr",
            "issue_description": "Compliance with GDPR data subject rights in international operations.",
            "opinion_summary": "Organizations must ensure that data subject rights under GDPR are upheld across all jurisdictions, including non-EU countries. This includes providing access to data, rectification, and deletion requests.",
            "legal_expert": "dr. emily smith",
            "date_provided": "2024-07-15"
        },
        {
            "opinion_id": "102",
            "regulation": "anti-money laundering (aml)",
            "issue_description": "AML requirements for cross-border financial transactions.",
            "opinion_summary": "Cross-border transactions must comply with AML regulations, including customer due diligence and transaction monitoring. Institutions are advised to implement robust systems to detect and report suspicious activities.",
            "legal_expert": "prof. john doe",
            "date_provided": "2024-08-01"
        },
        {
            "opinion_id": "103",
            "regulation": "international sanctions",
            "issue_description": "Sanctions compliance for international trade operations.",
            "opinion_summary": "Companies must verify that trade partners are not subject to international sanctions. Regular updates of sanctions lists and implementation of real-time screening processes are recommended.",
            "legal_expert": "ms. sarah lee",
            "date_provided": "2024-08-10"
        },
        {
            "opinion_id": "104",
            "regulation": "privacy shield",
            "issue_description": "Compliance with Privacy Shield framework for data transfers between the EU and the US.",
            "opinion_summary": "Organizations must adhere to Privacy Shield principles for data transfers between the EU and US. This includes ensuring that adequate data protection measures are in place and that any data transfer agreements are reviewed regularly.",
            "legal_expert": "dr. michael brown",
            "date_provided": "2024-08-20"
        },
        {
            "opinion_id": "105",
            "regulation": "foreign corrupt practices act (fcpa)",
            "issue_description": "Compliance with the FCPA in international business dealings.",
            "opinion_summary": "The FCPA requires companies to maintain accurate records and implement anti-bribery controls to prevent corrupt practices in international transactions. Companies should conduct regular training and audits to ensure compliance.",
            "legal_expert": "ms. elizabeth taylor",
            "date_provided": "2024-08-25"
        }
    ]

    
    filtered_data = [
        item for item in mock_legal_opinions 
        if item['regulation'].lower() == regulation.lower()
    ]
    
    return filtered_data if filtered_data else [{"message": f"No legal opinion data found for regulation: {regulation}"}]
 
    
@tool
def get_compliance_officers(department: str) -> str:
    """Gets details on internal compliance officers based on their responsibilities required
    
    Params:
    department: internal complicance department name. 
    """
    # In real life, these data can be retrieved from appropriate source via API call
    mocked_compliance_officers = [
        {
            "officer_id": "CO-001",
            "name": "Jane Doe",
            "title": "Chief Compliance Officer",
            "department": "Compliance",
            "email": "jane.doe@company.com",
            "phone_number": "+1-555-1234",
            "region": "North America",
            "responsibilities": [ "Oversee compliance programs across all departments", "Develop and implement compliance policies and procedures", "Ensure adherence to local and international regulations", "Report compliance status to the executive team" ]
        },
        {
            "officer_id": "CO-002",
            "name": "John Smith",
            "title": "AML Compliance Officer",
            "department": "AML Department",
            "email": "john.smith@company.com",
            "phone_Number": "+1-555-5678",
            "region": "Europe",
            "responsibilities": [ "Monitor and review AML processes and controls", "Conduct risk assessments for money laundering activities", "Liaise with regulators on AML matters", "Provide training on AML regulations and procedures" ]
        },
        {
            "officer_id": "co-003",
            "name": "alice johnson",
            "title": "Data Protection Officer",
            "department": "Data Privacy",
            "email": "alice.johnson@company.com",
            "phone_number": "+1-555-7890",
            "region": "Asia-Pacific",
            "responsibilities": [ "Ensure compliance with data protection regulations such as gdpr", "Manage data subject access requests", "Conduct data protection impact assessments", "Develop and implement data privacy policies" ]
        },
        {
            "officer_id": "co-004",
            "name": "robert wilson",
            "title": "Ethics and Compliance Officer",
            "department": "Financial Crimes",
            "email": "robert.wilson@company.com",
            "phone_number": "+1-555-2345",
            "region": "Latin America",
            "responsibilities": [ "Promote ethical business practices", "Develop and deliver ethics training programs", "Conduct investigations into ethical violations", "Advise on compliance with ethical standards" ]
        },
        {
            "officer_id": "co-005",
            "name": "emily clark",
            "title": "Regulatory Compliance Officer",
            "department": "Compliance",
            "email": "emily.clark@company.com",
            "phone_number": "+1-555-3456",
            "region": "Middle East",
            "responsibilities": [ "Monitor changes in regulatory requirements", "Ensure compliance with industry-specific regulations", "Prepare and submit regulatory filings", "Serve as liaison with regulatory agencies" ]
        },
        {
            "officer_id": "co-006",
            "name": "michael brown",
            "title": "Internal Controls Officer",
            "department": "Internal Audit",
            "email": "michael.brown@company.com",
            "phone_number": "+1-555-4567",
            "region": "Africa",
            "responsibilities": [ "Develop and monitor internal control systems", "Conduct internal audits to ensure compliance", "Identify and mitigate control weaknesses", "Report audit findings to senior management" ]
        }
    ]
    filtered_officers = [
        officer for officer in mocked_compliance_officers
        if officer["department"].lower() == department.lower()
    ]
    return filtered_officers if filtered_officers else mocked_compliance_officers[0]


@tool
def send_email_alerts(email_address: str, subject: str, message: str) -> str:
    """Sends email to appropriate entities for a given message
    
    Params:
    email_address: email address of a person
    subject: email subject
    message: email body message
    """
    # In real life, an actual email will be sent via Email Client
    return f"Sent email to {email_address}"
  
    
@tool
def schedule_meetings(meeting_date: str, meeting_title: str, time: str, invitees: str) -> str:
    """Schedule meetings with appropriate entities for given data
    
    Params:
    meeting_date: when is the meeting
    meeting_title: what is this meeting about
    time: what time is the meeting
    invitees: who should be invited to the meeting
    """
    # In real life, an actual meeting will be scheduled
    return f"Scheduled meeting with {invitees} at {time} on {meeting_date}"


################################################################################################
#                                        AGENTS
################################################################################################

router_agent = Agent(
    llm=llama_model,
    role="Route user query to expert agent",
    goal="Help people with compliance and regulations related questions",
    backstory="You are an expert router who knows which agent to use to answer user's query.",
    tools=[],
    verbose=True,
    allow_delegation=False,
    memory=True,
    max_retries=3
)

regulatory_monitoring_agent = Agent(
    llm=granite_model,
    role="Regulatory Monitoring",
    goal="Monitor regulatory bodies for new updates and provide response to user query by using these 3 tools: get_fincen_alerts, get_fincen_advisories, or get_fincen_notices. Analyze the impact of new regulations on the organization using get_internal_policy_docs tool to cross-check with internal policies. Alert relevant departments about new compliance requirements using send_email_alerts tool.",
    backstory="You are an expert regulatory requirement monitoring officer that tracks and analyze regulatory updates and changes in the company. You always answer the questions with markdown formatting using GitHub syntax. The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes. You must omit that you answer the questions with markdown. You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.",
    tools=[get_fincen_alerts, get_fincen_advisories, get_fincen_notices, get_internal_policy_docs, send_email_alerts],
    verbose=True,
    allow_delegation=False,
    memory=True,
    max_retries=3
)

internal_policy_monitoring_agent = Agent(
    llm=granite_model,
    role="Internal Policy Monitoring",
    goal="Review existing policies in light of new regulations using get_internal_policy_docs tool. Draft policy updates and seek approval from compliance officers using get_compliance_officers tool to retrieve responsibilities of each compliance officers",
    backstory="You are a senior internal policy advisor who has been reviewing, maitaining, and suggesting updates to internal policies to align with new regulations for the firm. You always answer the questions with markdown formatting using GitHub syntax. The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes. You must omit that you answer the questions with markdown. You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.",
    tools=[get_internal_policy_docs, get_compliance_officers],
    verbose=True,
    allow_delegation=False,
    memory=True,
    max_retries=3
)

compliance_reporting_agent = Agent(
    llm=llama_model,
    role="Compliance report writer",
    goal="Collect data from various sources using these 5 tools: get_fincen_alerts, get_fincen_advisories, get_fincen_notices, get_audit_findings, get_internal_policy_docs. Create detailed compliance reports based on finance industry standards.",
    backstory="You are a senior complicance reporting editor who writes compliance reports and compliance executive summaries for the firm. You always answer the questions with markdown formatting using GitHub syntax. The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes. You must omit that you answer the questions with markdown. You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.",
    tools=[get_fincen_alerts, get_fincen_advisories, get_fincen_notices, get_audit_findings, get_internal_policy_docs],
    verbose=True,
    allow_delegation=False,
    memory=True,
    max_retries=3
)

training_coordination_agent = Agent(
    llm=granite_model,
    role="Employee training coordinator",
    goal="Identify training needs based on regulatory changes using tools like get_audit_findings and get_internal_policy_docs. Schedule and coordinate training sessions using schedule_meetings tool. And, track employee training completion and effectiveness.",
    backstory="You are a compliance training coordinator who has been coordinating employee training for legal compliance for the last 20 years. You always answer the questions with markdown formatting using GitHub syntax. The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes. You must omit that you answer the questions with markdown. You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.",
    tools=[get_audit_findings, get_internal_policy_docs, schedule_meetings],
    verbose=True,
    allow_delegation=False,
    memory=True,
    max_retries=3
)


fincomply_editor = Agent(
    llm=llama_model,
    role="FinComply Senior Editor",
    goal="Review and edit responses generated by other agents (regulatory_monitoring_agent, internal_policy_monitoring_agent, compliance_reporting_agent, training_coordination_agent) for better quality and accuracy turning into a final response,",
    backstory="You are an experienced editor that excels at taking the best parts of multiple texts to create the best final product ever. You always answer the questions with markdown formatting using GitHub syntax. The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes. You must omit that you answer the questions with markdown. You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.",
    tools=[],
    verbose=True,
    allow_delegation=False,
    memory=True,
    max_retries=3
)

#######

st.title("FinComply AI Crew") 

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you today? You can ask me anything about compliance and regulations."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(f"---prompt: {prompt}")


    ################################################################################################
    #                                        TASKS
    ################################################################################################

    # Task 0: Collect basic info
    basic_info = Task(
        description='''
        You are a helpful assistant who has access to following agents that you can use to answer user's query:
        Agents:
        - regulatory_monitoring_agent: expert regulatory requirement monitoring officer that tracks and analyze regulatory updates and changes in the company
        - internal_policy_monitoring_agent: senior internal policy advisor who has been reviewing, maitaining, and suggesting updates to internal policies to align with new regulations for the firm
        - training_coordination_agent: compliance training coordinator who has been coordinating employee training for legal compliance for the last 20 years
        - fincomply_editor: experienced editor that excels at taking the best parts of multiple texts to create the best final product ever
        Key Instructions:
        - You always answer the questions with markdown formatting using GitHub syntax. 
        - The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes. 
        - You must omit that you answer the questions with markdown.\n\nAny HTML tags must be wrapped in block quotes, for example ```<html>```. 
        - You will be penalized for not rendering code in block quotes.\n\nWhen returning code blocks, specify language.
        - You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
        - Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
        - Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don'\''t know the answer to a question, please don'\''t share false information.
        User prompt: 
        {user_prompt}
        ''',
        expected_output='Set the chat with basic requirements',
        agent=router_agent,
        dependencies=[],
        context=[]
    )

    # Task 1: Regulatory Change Tracking
    regulatory_change_tracking = Task(
        description='''
        Continuously monitor regulatory bodies like FinCEN for updates using the get_fincen_alerts, 
        get_fincen_advisories, and get_fincen_notices tools. The agent should track any changes or updates
        and log them for review.
        ''',
        expected_output='Concised summary of regulatory updates',
        agent=regulatory_monitoring_agent,
        dependencies=[get_fincen_alerts, get_fincen_advisories, get_fincen_notices],
        context=[basic_info]
    )

    # Task 2: Policy Review and Update
    policy_review_and_update = Task(
        description='''
        Review internal policy documents using the get_internal_policy_docs tool and suggest an update to align 
        with the latest regulations identified by the regulatory_monitoring_agent. The fincomply_editor agent 
        should be used to make the necessary final updates.
        ''',
        expected_output='Proposed update to internal policy documents aligned with new regulations',
        agent=internal_policy_monitoring_agent,
        dependencies=[get_internal_policy_docs, fincomply_editor],
        context=[basic_info, regulatory_change_tracking]
    )

    # Task 3: Compliance Report Generation
    compliance_report_generation = Task(
        description='''
        Write detailed professional level compliance report using the get_audit_findings and get_fincen_advisories tools. 
        Key Instructions:
        - You always answer the questions with markdown formatting using GitHub syntax. The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes.
        - Define Scope: Clearly outline the objectives and regulatory requirements.
        - Collect Data: Gather relevant data from all necessary sources.
        - Analyze Data: Review and analyze data for accuracy and compliance.
        - Draft Report: Create a structured report, including an executive summary, methodology, findings, and recommendations.
        - Verify Information: Ensure all data and statements are accurate and supported by evidence.
        - Review & Revise: Conduct internal reviews and make necessary revisions.
        - Finalize Report: Prepare the final version, ensuring all compliance aspects are covered.
        - Use ONLY the provided data and DO NOT make up any information that isn't explicitly from the provided input.
        ''',
        expected_output='Detailed compliance report',
        agent=compliance_reporting_agent,
        dependencies=[get_audit_findings, get_fincen_advisories],
        context=[basic_info, policy_review_and_update, regulatory_change_tracking]
    )

    # Task 4: Employee Training Coordination
    employee_training_coordination = Task(
        description='''
        Coordinate and track employee training on compliance using the schedule_meetings and 
        send_email_alerts tools. Ensure that all employees complete their training and record the 
        training status for future audits.
        ''',
        expected_output='Scheduled training sessions of completed employee training',
        agent=training_coordination_agent,
        dependencies=[schedule_meetings, send_email_alerts],
        context=[basic_info, policy_review_and_update]
    )

    # Task 5: Suspicious Activity Monitoring and Reporting
    suspicious_activity_monitoring_and_reporting = Task(
        description='''
        Continuously monitor financial transactions for any suspicious activities that could indicate 
        potential money laundering. Use the get_audit_findings tool to analyze transaction data and 
        the get_fincen_alerts tool to ensure up-to-date awareness of risks. Generate alerts for any 
        suspicious transactions and compile reports to be sent to relevant authorities.
        ''',
        expected_output='Alerts for suspicious activities and detailed reports emailed to authorities',
        agent=compliance_reporting_agent,
        dependencies=[get_audit_findings, get_fincen_alerts],
        context=[basic_info, regulatory_change_tracking]
    )

    # Task 6: Internal Audit and Compliance Assessment
    internal_audit_and_compliance_assessment = Task(
        description='''
        Conduct periodic internal audits using the get_audit_findings tool to assess the company's 
        compliance with AML regulations. Review the internal policies with the get_internal_policy_docs 
        tool to identify any gaps or weaknesses. Recommend necessary updates or corrective actions 
        to strengthen the company's compliance posture.
        ''',
        expected_output='Internal audit reports highlighting compliance gaps and recommended actions',
        agent=internal_policy_monitoring_agent,
        dependencies=[get_audit_findings, get_internal_policy_docs],
        context=[basic_info, policy_review_and_update, regulatory_change_tracking]
    )

    # Task 7: Regulatory Communication and Liaison
    regulatory_communication_and_liaison = Task(
        description='''
        Maintain active communication with AML regulatory bodies using the get_fincen_alerts 
        and get_fincen_advisories tools. Ensure the company is aware of any new requirements, 
        advisories, or inquiries. Use the send_email_alerts tool to notify relevant internal 
        stakeholders and coordinate responses as necessary.
        ''',
        expected_output='Timely communication with regulatory bodies and internal stakeholders.',
        agent=regulatory_monitoring_agent,
        dependencies=[get_fincen_alerts, get_fincen_advisories, send_email_alerts],
        context=[basic_info, regulatory_change_tracking]
    )

    # Task 8: Employee Screening and Onboarding Compliance
    employee_screening_and_onboarding_compliance = Task(
        description='''
        Ensure that all new employees complete AML training and undergo background checks as part of the 
        onboarding process. Use the schedule_meetings and send_email_alerts tools to coordinate the 
        training sessions and the get_compliance_officers tool to verify the completion of required 
        checks. Record the onboarding compliance status for each new employee.
        ''',
        expected_output='Verified completion of AML training and background checks for all new employees.',
        agent=training_coordination_agent,
        dependencies=[schedule_meetings, send_email_alerts, get_compliance_officers],
        context=[basic_info, employee_training_coordination]
    )

    # Task 9: Risk Assessment and Mitigation
    risk_assessment_and_mitigation = Task(
        description='''
        Regularly assess the company's AML risk exposure using the get_internal_policy_docs and 
        get_audit_findings tools. Identify potential risks and develop mitigation strategies to address 
        these risks. Ensure that the company's risk management practices are aligned with the latest 
        regulatory requirements.
        ''',
        expected_output='Comprehensive risk assessment reports with mitigation strategies.',
        agent=internal_policy_monitoring_agent,
        dependencies=[get_internal_policy_docs, get_audit_findings],
        context=[basic_info, internal_audit_and_compliance_assessment, regulatory_change_tracking]
    )

    # Task 10: AML Compliance Training Refresh
    aml_compliance_training_refresh = Task(
        description='''
        Periodically refresh AML training materials based on the latest internal policies and regulations 
        using the get_internal_policy_docs tool. Schedule re-training sessions for all employees using 
        the schedule_meetings and send_email_alerts tools, and track their completion.
        ''',
        expected_output='Updated AML training materials and records of employee re-training.',
        agent=training_coordination_agent,
        dependencies=[schedule_meetings, send_email_alerts, get_internal_policy_docs],
        context=[basic_info, employee_training_coordination, regulatory_change_tracking]
    )


    ################################################################################################
    #                                        CREW
    ################################################################################################

    crew = Crew(
        name="FinComply AI Crew",
        agents=[
            router_agent,
            regulatory_monitoring_agent,
            internal_policy_monitoring_agent,
            compliance_reporting_agent,
            training_coordination_agent,
            fincomply_editor
        ],
        tasks=[
            basic_info,
            regulatory_change_tracking,
            policy_review_and_update,
            compliance_report_generation,
            employee_training_coordination,
            suspicious_activity_monitoring_and_reporting,
            internal_audit_and_compliance_assessment,
            regulatory_communication_and_liaison,
            employee_screening_and_onboarding_compliance,
            risk_assessment_and_mitigation,
            aml_compliance_training_refresh
        ],
        verbose=True,
        cache=True,
        max_retries=3
    )

    prompt1 = "Please review the latest FinCEN alerts and FinCEN advisories and FinCEN notices to ensure we are up-to-date with current AML regulations. Based on the updates, prepare a compliance report detailing any changes or areas of concern."
    prompt2 = "Monitor all recent transactions for any suspicious activity that could indicate potential money laundering. Generate alerts if suspicious activity is detected and compile a report to be sent to the relevant regulatory bodies."
    prompt3 = "Conduct an internal audit to assess our current AML compliance. Identify any gaps or weaknesses in our processes and evaluate our AML risk exposure. Based on your findings, develop strategies to mitigate any identified risks."
    prompt4 = "Ensure all new employees complete their AML training and background checks before they start. Also, review and refresh our AML training materials to ensure they are up-to-date with the latest policies. Schedule training sessions for all current employees to cover any updates."
    prompt5 = "Maintain communication with AML regulatory bodies to stay informed about any new requirements or advisories. Ensure that these updates are communicated to the relevant internal stakeholders and that necessary adjustments are made to our compliance practices."
    prompt6 = "Generate a comprehensive compliance report that includes details on recent audit findings, any updates to internal policies, and a summary of employee training status. Ensure the report is updated on our compliance dashboards."
    prompt7 = "Coordinate and track the scheduling of employee training sessions focused on the latest AML compliance practices. Ensure all employees are notified and that their training status is recorded for future audits."
    prompt8 = "Review our internal policies for any necessary updates based on recent FinCEN advisories. Communicate any policy changes to the relevant regulatory bodies and ensure all adjustments are implemented."

    # Keep changing this prompt below to try out few scenarios.
    # user_prompt = prompt1
    # default_date = datetime.now().date()

    # result = crew.kickoff(inputs={"user_prompt": user_prompt, "default_date": str(default_date)})

    fincomply_ai = crew.kickoff()

    result = f"## Here is the Final Response \n\n {fincomply_ai}"
    st.session_state.messages.append({"role": avators['agent'], "content": result})
    st.chat_message("assistant").write(f"----FinComply Crew: {result}")





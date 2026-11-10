import re

# Extend this list for your domain. Matching is case-insensitive, whole-term.
SKILLS = [
    "python", "java", "c++", "c#", "javascript", "typescript", "sql", "mysql", "postgresql",
    "mongodb", "html", "css", "react", "angular", "node.js", "django", "flask", "spring",
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "keras",
    "pytorch", "scikit-learn", "pandas", "numpy", "matplotlib", "tableau", "power bi", "excel",
    "statistics", "data analysis", "data visualization", "hadoop", "spark", "aws", "azure",
    "gcp", "docker", "kubernetes", "jenkins", "git", "linux", "ci/cd", "selenium", "junit",
    "manual testing", "automation testing", "rest api", "microservices", "agile", "scrum",
    "recruitment", "payroll", "employee relations", "talent acquisition", "sap", "autocad",
    "civil engineering", "project management", "networking", "cybersecurity", "blockchain",
    # business / HR / sales / finance
    "sales", "business development", "crm", "salesforce", "marketing", "seo", "accounting",
    "tally", "financial analysis", "budgeting", "auditing", "customer service", "negotiation",
    "leadership", "communication", "operations", "supply chain", "logistics",
    # marketing
    "social media", "email marketing", "lead generation", "content marketing",
    "digital marketing", "content planning", "campaign management", "paid advertising",
    "google analytics", "brand management", "market research", "copywriting",
    # engineering / other domains
    "solidworks", "matlab", "plc", "scada", "cad", "embedded systems", "iot", "vlsi",
    "etl", "data warehouse", "airflow", "power query", "devops", "terraform", "ansible",
    "web development", "android", "ios", "swift", "kotlin", "php", "ruby",
    "legal research", "litigation", "contract drafting", "nutrition", "fitness training",
    "teaching", "curriculum development", "photoshop", "illustrator", "video editing",
]


def extract_skills(text: str) -> set:
    text = text.lower()
    found = set()
    for skill in SKILLS:
        pattern = r"(?<![a-z0-9+#])" + re.escape(skill) + r"s?(?![a-z0-9+#])"
        if re.search(pattern, text):
            found.add(skill)
    return found

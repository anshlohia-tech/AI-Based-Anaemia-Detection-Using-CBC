import re


def find_value(text, keywords):
    """
    Search for a keyword and return the first numeric value nearby.
    """

    for keyword in keywords:

        pattern = rf"{keyword}[^\d]*(\d+\.?\d*)"

        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            try:
                return float(match.group(1))
            except:
                return None

    return None


def extract_cbc_values(text):

    values = {

        "HGB": find_value(text, [
            "Haemoglobin",
            "Hemoglobin",
            "HGB"
        ]),

        "WBC": find_value(text, [
            "Total Count",
            "WBC",
            "Total Count (WBC)"
        ]),

        "NEUTp": find_value(text, [
            "Neutrophils"
        ]),

        "LYMp": find_value(text, [
            "Lymphocytes"
        ]),

        "MONOp": find_value(text, [
            "Monocytes"
        ]),

        "EOSp": find_value(text, [
            "Eosinophils"
        ]),

        "BASOp": find_value(text, [
            "Basophils"
        ]),

        "PLT": find_value(text, [
            "Platelets"
        ]),

        "RBC": find_value(text, [
            "RBC"
        ]),

        "HCT": find_value(text, [
            "PCV",
            "HCT",
            "Hematocrit"
        ]),

        "MCV": find_value(text, [
            "MCV"
        ]),

        "MCH": find_value(text, [
            "MCH"
        ]),

        "MCHC": find_value(text, [
            "MCHC"
        ])

    }

    return values
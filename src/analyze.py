# 13/11/18 E.C 12:35 in the moring.....
# Step 3 - Analyze (Domain Classification + EDA)

import logging
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------- Domain Classification --------------

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "Religious": 
    ["god", "jesus", "christ", "lord", "bible", "scripture", "church",
        "holy", "spirit", "prayer", "pray", "heaven", "sin", "faith", "apostle",
        "gospel", "prophet", "disciple", "salvation", "sacred", "kingdom of god",
        "jehovah", "angel", "satan", "paul", "moses", "israelite", "temple",
        "worship", "psalm", "covenant"],
    "Government": 
    ["minister", "government", "parliament", "president", "policy",
        "ministry", "cabinet", "election", "constitution", "prime minister",
        "official", "administration", "diplomat", "embassy", "sovereign",
        "regime", "federal", "republic"],
    "News": 
    ["reported", "according to", "sources say", "announced", "breaking",
        "journalist", "press release", "spokesperson", "yesterday", "today",
        "this week", "witnesses", "correspondent"],
    "Legal": 
    ["court", "judge", "lawsuit", "plaintiff", "defendant", "verdict",
        "legal", "law", "attorney", "prosecutor", "contract", "regulation",
        "statute", "jurisdiction"],
    "Medical": 
    ["doctor", "hospital", "patient", "disease", "treatment", "symptom",
        "medicine", "diagnosis", "surgery", "vaccine", "clinic", "nurse",
        "infection", "therapy"],
    "Educational":
      ["student", "school", "university", "teacher", "lesson", "study",
        "exam", "classroom", "curriculum", "professor", "degree", "lecture"],
    "Conversational":
      ["i'm", "you're", "hey", "hi ", "thanks", "please", "sorry",
        "how are you", "let's", "okay", "yeah", "gonna", "wanna"],
}


def score_domains(text):

    text = str(text).lower()

    scores = Counter()

    for domain, words in DOMAIN_KEYWORDS.items():

        for word in words:

            if word in text:
                scores[domain] += 1

    return scores



def classify_domain(text, source):

    scores = score_domains(text)


    if scores:
        return scores.most_common(1)[0][0]


    source = str(source).lower()


    if "parallel" in source:
        return "News"

    if "custom" in source:
        return "Conversational"


    return "General"



def add_domain_column(df):

    result = df.copy()

    result["domain"] = result.apply(
        lambda row:
        classify_domain(
            row["english"],
            row["source_file"]
        ),
        axis=1
    )

    return result



def domain_distribution(df):

    counts = df["domain"].value_counts()

    result = pd.DataFrame({

        "domain": counts.index,

        "count": counts.values,

        "percentage":
        (counts.values / len(df) * 100).round(2)

    })

    return result

def plot_domain_distribution(dist, plots_dir):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8,5))
    plt.bar(dist["domain"], dist["count"])
    plt.xticks(rotation=45)

    plt.title("Dataset Domain Distribution")
    plt.xlabel("Domain")
    plt.ylabel("Number of Sentence Pairs")

    plt.tight_layout()

    plt.savefig(
        plots_dir / "domain_distribution.png",
        dpi=120
    )

    plt.close()




# ---------------- EDA ----------------


WORD_RE = re.compile(r"\S+")


def tokenize_words(text):

    return WORD_RE.findall(str(text))



def compute_eda_stats(df):

    am_length = df["amharic"].apply(
        lambda x: len(tokenize_words(x))
    )

    en_length = df["english"].apply(
        lambda x: len(tokenize_words(x))
    )


    am_vocab = Counter()

    en_vocab = Counter()


    for sentence in df["amharic"]:

        am_vocab.update(
            tokenize_words(sentence)
        )


    for sentence in df["english"]:

        en_vocab.update(
            word.lower()
            for word in tokenize_words(sentence)
        )


    return {


        "pairs":len(df),


        "am_length":
        am_length.describe().to_dict(),


        "en_length":
        en_length.describe().to_dict(),


        "am_vocab":
        len(am_vocab),


        "en_vocab":
        len(en_vocab),


        "am_top":
        am_vocab.most_common(20),


        "en_top":
        en_vocab.most_common(20),


        "am_lengths":
        am_length,


        "en_lengths":
        en_length

    }



# ---------------- Plot Generation ----------------


def generate_plots(stats, output):

    output = Path(output)

    output.mkdir(
        parents=True,
        exist_ok=True
    )


    # sentence length plot

    plt.figure(figsize=(8,4))

    plt.hist(
        stats["am_lengths"],
        bins=50
    )

    plt.title(
        "Amharic Sentence Length"
    )

    plt.xlabel("Words")

    plt.ylabel("Frequency")


    plt.savefig(
        output / "amharic_length.png"
    )

    plt.close()



    plt.figure(figsize=(8,4))


    plt.hist(
        stats["en_lengths"],
        bins=50
    )


    plt.title(
        "English Sentence Length"
    )


    plt.xlabel("Words")

    plt.ylabel("Frequency")


    plt.savefig(
        output / "english_length.png"
    )


    plt.close()



# ---------------- Report ----------------


def step3_report(dist, stats):# domain distrbution

    text=[]


    text.append(
        "# Step 3 Analysis Report\n"
    )


    text.append(
        "## Domain Distribution\n"
    )


    text.append(
        str(dist)
    )


    text.append(
        f"""
Corpus size:
{stats['pairs']}

Amharic vocabulary:
{stats['am_vocab']}

English vocabulary:
{stats['en_vocab']}

"""
    )


    return "\n".join(text)






if __name__ == "__main__":

    INPUT = Path(
        "data/processed/cleaned_dataset.csv"
    )

    OUTPUT = Path(
        "outputs"
    )

    # Create output folders
    (OUTPUT / "plots").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT / "reports").mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading cleaned data...")

    df = pd.read_csv(INPUT)

    print("Adding domain labels...")

    df = add_domain_column(df)

    # Domain distribution
    dist = domain_distribution(df)

    print(dist)

    # Save domain distribution graph
    plot_domain_distribution(
        dist,
        OUTPUT / "plots"
    )

    # EDA analysis
    stats = compute_eda_stats(df)

    # Save EDA plots
    generate_plots(
        stats,
        OUTPUT / "plots"
    )

    # Create report
    report = step3_report(
        dist,
        stats
    )

    # Save report
    with open(
        OUTPUT / "reports/step3_report.md",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    # Create processed data folder
    PROCESSED = Path("data/processed")

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save analyzed dataset
    df.to_csv(
        PROCESSED / "analyzed_dataset.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(stats["am_length"])
    print(stats["en_length"])

    print("Step 3 completed successfully!")
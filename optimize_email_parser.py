import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from textwrap import indent

import dspy

from dotenv import load_dotenv

from yaml_formatter import YAMLFormatter

BASE_DIR = Path(__file__).parent
MODEL = os.getenv("DSPY_MODEL", "anthropic/claude-haiku-4-5-20251001")

load_dotenv(BASE_DIR / ".env")

log_file = BASE_DIR / "lm_prompts.log"
file_handler = RotatingFileHandler(log_file)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

logger = logging.getLogger("Prompt Logger")
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

yaml_formatter = YAMLFormatter()


class LoggingLM(dspy.LM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, prompt=None, messages=None, **kwargs):
        results = super().forward(prompt, messages, **kwargs)
        if prompt is not None:
            logger.info(f"Prompt: \n" + indent(str(prompt), prefix="   "))
        if messages is not None:
            messages_txt = indent(yaml_formatter.format(messages), prefix="   ")
            logger.info(f"Messages: \n" + messages_txt)
            completions_txt = indent(
                yaml_formatter.format([str(choice.message.content) for choice in results.choices]), prefix="   "
            )
            logger.info(f"Completion: \n" + completions_txt)

        return results


lm = LoggingLM(MODEL, temperature=0.1, max_tokens=1000)

dspy.configure(lm=lm)


def load_training_data(data_dir=None):
    if data_dir is None:
        data_dir = BASE_DIR / "data"
    data_dir = Path(data_dir)

    examples = []
    email_files = sorted(data_dir.glob("email_*.txt"))

    for email_path in email_files:
        num = email_path.stem.split("_")[1]
        json_path = data_dir / f"data_{num}.json"

        email_content = email_path.read_text().strip()
        extraction_data = json.loads(json_path.read_text())

        example = dspy.Example(
            email=email_content,
            intent=extraction_data["intent"],
            action_items="; ".join(extraction_data["action_items"]),
            deadlines="; ".join(extraction_data["deadlines"]),
            priority=extraction_data["priority"],
        ).with_inputs("email")

        examples.append(example)

    return examples


# DSPy Signature for email parsing
class EmailToExtraction(dspy.Signature):
    """Extract structured information from an email including intent, action items, deadlines, and priority."""

    email = dspy.InputField(desc="The email content to parse")
    intent = dspy.OutputField(desc="Primary intent: task assignment, meeting request, FYI, etc.")
    action_items = dspy.OutputField(desc="Semicolon-separated list of action items with assignees")
    deadlines = dspy.OutputField(desc="Semicolon-separated list of deadlines or dates mentioned")
    priority = dspy.OutputField(desc="Priority level: high, medium, low")


# DSPy Module for email parsing
class EmailParser(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(EmailToExtraction)

    def forward(self, email):
        return self.extract(email=email)


# LLM-as-judge evaluator signature
class ExtractionEvaluator(dspy.Signature):
    """Evaluate how well the predicted extraction matches the ground truth extraction."""

    email = dspy.InputField(desc="Original email content")
    predicted_intent = dspy.InputField(desc="Predicted intent")
    predicted_action_items = dspy.InputField(desc="Predicted action items")
    predicted_deadlines = dspy.InputField(desc="Predicted deadlines")
    predicted_priority = dspy.InputField(desc="Predicted priority")
    true_intent = dspy.InputField(desc="Ground truth intent")
    true_action_items = dspy.InputField(desc="Ground truth action items")
    true_deadlines = dspy.InputField(desc="Ground truth deadlines")
    true_priority = dspy.InputField(desc="Ground truth priority")

    intent_score = dspy.OutputField(desc="Score for intent accuracy (0-1)")
    action_items_score = dspy.OutputField(desc="Score for action items accuracy (0-1)")
    deadlines_score = dspy.OutputField(desc="Score for deadlines accuracy (0-1)")
    priority_score = dspy.OutputField(desc="Score for priority accuracy (0-1)")
    reasoning = dspy.OutputField(desc="Brief explanation of the scores")


# Create LLM-as-judge evaluator
class LLMJudge(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.ChainOfThought(ExtractionEvaluator)

    def forward(self, email, pred, true):
        result = self.evaluate(
            email=email,
            predicted_intent=pred.intent,
            predicted_action_items=pred.action_items,
            predicted_deadlines=pred.deadlines,
            predicted_priority=pred.priority,
            true_intent=true.intent,
            true_action_items=true.action_items,
            true_deadlines=true.deadlines,
            true_priority=true.priority,
        )
        return result


judge = LLMJudge()


# Evaluation metric using LLM as judge
def extraction_accuracy(example, pred, trace=None):
    """Calculate accuracy of extraction using LLM as judge"""

    def parse_score(value, default=0.0):
        try:
            score = float(value)
            return max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            return default

    try:
        evaluation = judge(email=example.email, pred=pred, true=example)

        scores = [
            parse_score(evaluation.intent_score),
            parse_score(evaluation.action_items_score),
            parse_score(evaluation.deadlines_score),
            parse_score(evaluation.priority_score),
        ]

        return sum(scores) / len(scores)

    except Exception as e:
        print(f"Error in evaluation: {e}")
        return 0.0


# Optimize the email parser
def optimize_email_parser(train_examples):
    """Optimize the email parser using DSPy"""

    parser = EmailParser()

    # Other optimizers to try: dspy.BootstrapFewShot, dspy.MIPROv2
    # See https://dspy.ai/api/category/optimizers/
    teleprompter = dspy.COPRO(
        metric=extraction_accuracy,
        breadth=3,  # Try more instruction variants
        depth=3,  # More optimization rounds
        init_temperature=1.2,
        verbose=False,
        track_stats=True,  # See what COPRO learned
    )

    return teleprompter.compile(parser, trainset=train_examples, eval_kwargs={})


# Function to test the parser
def test_parser(parser, test_example):
    """Test the parser on a single example"""
    prediction = parser(email=test_example.email)

    # Calculate score if ground truth is available
    if hasattr(test_example, "intent"):
        score = extraction_accuracy(test_example, prediction)
        print(f"\nAccuracy Score: {score:.2%}")

    return prediction


def inspect_parser(parser):
    """Print what DSPy learned during optimization."""
    print("\n" + "=" * 60)
    print("PROMPT STRUCTURE:")
    print("=" * 60)

    if hasattr(parser.extract, "demos") and parser.extract.demos:
        print(f"\nDSPy selected {len(parser.extract.demos)} demonstrations")
        print("\nExample demonstrations:")
        for i, demo in enumerate(parser.extract.demos[:2]):
            print(f"\nDemo {i + 1}:")
            print(f"Email: {demo.email[:150]}...")
            print(f"Intent: {demo.intent}")
            print(f"Action Items: {demo.action_items}")
            print(f"Priority: {demo.priority}")

    # Access signature through the predict attribute
    if hasattr(parser.extract, "predict") and hasattr(parser.extract.predict, "signature"):
        print(f"\nSignature: {parser.extract.predict.signature}")
    elif hasattr(parser.extract, "signature"):
        print(f"\nSignature: {parser.extract.signature}")
    else:
        print("\nSignature: Not directly accessible")
        print(f"Available attributes on extract: {[attr for attr in dir(parser.extract) if not attr.startswith('_')]}")

    if hasattr(parser.extract, "extended_signature"):
        print(f"\nExtended signature:")
        print(parser.extract.extended_signature)
    elif hasattr(parser.extract, "predict") and hasattr(parser.extract.predict, "extended_signature"):
        print(f"\nExtended signature:")
        print(parser.extract.predict.extended_signature)

    if hasattr(parser.extract, "predict") and hasattr(parser.extract.predict, "lm"):
        print("\nOptimized prompt structure created by DSPy")


def run_demo():
    print("Loading training data...")
    examples = load_training_data()
    print(f"Loaded {len(examples)} training examples")

    print("\nTraining DSPy email parser...")
    optimized_parser = optimize_email_parser(examples)

    inspect_parser(optimized_parser)

    return optimized_parser


if __name__ == "__main__":
    optimized_parser = run_demo()
    optimized_parser.save(BASE_DIR / "optimized_parser.json")

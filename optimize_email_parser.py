import json
import logging
import os
from logging.handlers import RotatingFileHandler
from textwrap import indent

import dspy

from dotenv import load_dotenv

from yaml_formatter import YAMLFormatter

load_dotenv("./.env")

log_file = 'lm_prompts.log'
file_handler = RotatingFileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

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
            logger.info(f"Prompt: \n" + indent(str(prompt), prefix='   '))
        if messages is not None:
            messages_txt = indent(yaml_formatter.format(messages), prefix='   ')
            logger.info(f"Messages: \n" + messages_txt)
            completions_txt = indent(yaml_formatter.format([str(choice.message.content) for choice in results.choices]), prefix='   ')
            logger.info(f"Completion: \n" + completions_txt)

        return results


lm = LoggingLM(
    'openai/gpt-3.5-turbo',
    temperature=0.1,
    max_tokens=1000
)

dspy.settings.configure(lm=lm)


def load_training_data(data_dir="./data/"):
    examples = []
    email_files = sorted([f for f in os.listdir(data_dir) if f.startswith("email_") and f.endswith(".txt")])

    for email_file in email_files:
        num = email_file.split("_")[1].split(".")[0]
        json_file = f"data_{num}.json"

        with open(os.path.join(data_dir, email_file), 'r') as f:
            email_content = f.read().strip()

        with open(os.path.join(data_dir, json_file), 'r') as f:
            extraction_data = json.load(f)

        example = dspy.Example(
            email=email_content,
            intent=extraction_data['intent'],
            action_items='; '.join(extraction_data['action_items']),
            deadlines='; '.join(extraction_data['deadlines']),
            priority=extraction_data['priority']
        ).with_inputs('email')

        examples.append(example)

    return examples


# A simple baseline prompt (for comparison)
class BaselineEmailParser:
    def __init__(self):
        self.prompt_template = """Extract the following from this email:
- Intent (task assignment, meeting request, FYI, etc.)
- Action items with assignees
- Deadlines
- Priority level

Email: {email}

Please extract in JSON format."""

    def __call__(self, email):
        class BaselinePrediction:
            def __init__(self):
                self.intent = "unknown"
                self.action_items = "Extract manually"
                self.deadlines = "Check email"
                self.priority = "medium"

        return BaselinePrediction()


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
            true_priority=true.priority
        )
        return result


judge = LLMJudge()


# Evaluation metric using LLM as judge
def extraction_accuracy(example, pred, trace=None):
    """Calculate accuracy of extraction using LLM as judge"""
    try:
        evaluation = judge(
            email=example.email,
            pred=pred,
            true=example
        )

        scores = [
            float(evaluation.intent_score),
            float(evaluation.action_items_score),
            float(evaluation.deadlines_score),
            float(evaluation.priority_score)
        ]

        avg_score = sum(scores) / len(scores)
        return avg_score

    except Exception as e:
        print(f"Error in evaluation: {e}")
        return 0.25  # Basic score if evaluation fails


# Optimize the email parser
def optimize_email_parser(train_examples):
    """Optimize the email parser using DSPy"""

    # Create base parser
    parser = EmailParser()

    # Create the teleprompter (optimizer)
    # teleprompter = dspy.BootstrapFewShot(
    #     metric=extraction_accuracy,
    #     max_bootstrapped_demos=4, # How many examples to include in the prompt
    #     max_labeled_demos=4, # How many labeled examples to use
    #     max_rounds=1, # Number of optimization rounds (1 for demo speed)
    #     max_errors=5, # Maximum errors before giving up
    # )

    # return teleprompter.compile(
    #     parser,
    #     trainset=train_examples
    # )

    # https://dspy.ai/api/optimizers/MIPROv2/
    # teleprompter = dspy.MIPROv2(
    #     auto=None,
    #     metric=extraction_accuracy,
    #     num_candidates=4,  # Number of different signature variations
    #     init_temperature=0.7,  # Higher temp = more creative variations
    #     verbose=True
    # )
    # return teleprompter.compile(
    #     parser,
    #     trainset=train_examples[:-3],
    #     valset=train_examples[-3:],
    #     num_trials=2
    # )

    teleprompter = dspy.COPRO(
        metric=extraction_accuracy,
        breadth=3,  # Try more instruction variants
        depth=3,  # More optimization rounds
        init_temperature=1.2,
        verbose=False,
        track_stats=True,  # See what COPRO learned
    )

    return teleprompter.compile(
        parser,
        trainset=train_examples,
        eval_kwargs={}
    )


# Function to test the parser
def test_parser(parser, test_example):
    """Test the parser on a single example"""
    prediction = parser(email=test_example.email)

    # Calculate score if ground truth is available
    if hasattr(test_example, 'intent'):
        score = extraction_accuracy(test_example, prediction)
        print(f"\nAccuracy Score: {score:.2%}")

    return prediction


# Inspect optimized prompts
def inspect_parser(parser):
    print("\n" + "=" * 60)
    print("PROMPT STRUCTURE:")
    print("=" * 60)

    if hasattr(parser.extract, 'demos') and parser.extract.demos:
        print(f"\nDSPy selected {len(parser.extract.demos)} demonstrations")
        print("\nExample demonstrations:")
        for i, demo in enumerate(parser.extract.demos[:2]):
            print(f"\nDemo {i + 1}:")
            print(f"Email: {demo.email[:150]}...")
            print(f"Intent: {demo.intent}")
            print(f"Action Items: {demo.action_items}")
            print(f"Priority: {demo.priority}")

    # Access signature through the predict attribute
    if hasattr(parser.extract, 'predict') and hasattr(parser.extract.predict, 'signature'):
        print(f"\nSignature: {parser.extract.predict.signature}")
    elif hasattr(parser.extract, 'signature'):
        print(f"\nSignature: {parser.extract.signature}")
    else:
        print("\nSignature: Not directly accessible")
        print(
            f"Available attributes on extract: {[attr for attr in dir(parser.extract) if not attr.startswith('_')]}")

    if hasattr(parser.extract, 'extended_signature'):
        print(f"\nExtended signature:")
        print(parser.extract.extended_signature)
    elif hasattr(parser.extract, 'predict') and hasattr(parser.extract.predict,
                                                        'extended_signature'):
        print(f"\nExtended signature:")
        print(parser.extract.predict.extended_signature)

    if hasattr(parser.extract, 'predict') and hasattr(parser.extract.predict, 'lm'):
        print("\nOptimized prompt structure created by DSPy")


# Compare baseline vs optimized
def compare_parsers(samples, baseline_parser, optimized_parser):
    """Compare performance of baseline vs optimized parser"""
    results = []

    for i, example in enumerate(samples):
        print(f"\n{'=' * 60}")
        print(f"Test Email {i + 1}")
        print(f"{'=' * 60}")

        # Baseline prediction
        baseline_pred = baseline_parser(email=example.email)
        baseline_score = extraction_accuracy(example, baseline_pred)

        # Optimized prediction
        optimized_pred = optimized_parser(email=example.email)
        optimized_score = extraction_accuracy(example, optimized_pred)

        results.append({
            'email_num': i + 1,
            'baseline_score': baseline_score,
            'optimized_score': optimized_score,
            'improvement': optimized_score - baseline_score
        })

        print(f"\nBaseline Score: {baseline_score:.2%}")
        print(f"Optimized Score: {optimized_score:.2%}")
        print(f"Improvement: {optimized_score - baseline_score:.2%}")

    # Summary statistics
    avg_baseline = sum(r['baseline_score'] for r in results) / len(results)
    avg_optimized = sum(r['optimized_score'] for r in results) / len(results)

    print(f"\n{'=' * 60}")
    print("OVERALL PERFORMANCE:")
    print(f"{'=' * 60}")
    print(f"Average Baseline Score: {avg_baseline:.2%}")
    print(f"Average Optimized Score: {avg_optimized:.2%}")
    print(f"Average Improvement: {avg_optimized - avg_baseline:.2%}")

    return results


def run_demo():
    print("Loading training data...")
    examples = load_training_data()
    print(f"Loaded {len(examples)} training examples")

    print("\nTraining DSPy email parser...")
    optimized_parser = optimize_email_parser(examples)

    # print("\nComparing baseline vs DSPy optimized parser...")
    # baseline = BaselineEmailParser()

    # results = compare_parsers(examples, baseline, optimized_parser)
    #
    # if results:
    #     df = pd.DataFrame(results)
    #     print("\n" + "=" * 60)
    #     print("PERFORMANCE COMPARISON")
    #     print("=" * 60)
    #     print(f"Baseline Average Score: {df['baseline_score'].mean():.2%}")
    #     print(f"DSPy Optimized Score: {df['optimized_score'].mean():.2%}")
    #     print(f"Improvement: {(df['optimized_score'].mean() - df['baseline_score'].mean()):.2%}")

    # Show what DSPy learned
    inspect_parser(optimized_parser)

    return optimized_parser


if __name__ == "__main__":
    optimized_parser = run_demo()
    optimized_parser.save("optimized_parser.json")

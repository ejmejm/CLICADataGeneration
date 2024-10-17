import concurrent.futures
import json

from human_eval.data import read_problems
from human_eval.execution import check_correctness
import litellm
from pydantic import BaseModel


LLM_NAME = "gpt-4o-mini"
# "gpt-4o-mini", "gpt-4o", "o1-preview", "o1-mini"


class BugResponse(BaseModel):
    bug_explanation: str
    buggy_code: str


def introduce_bug(problem):
    prompt = f"""
    Here's a Python function:

    {problem['prompt']}
    {problem['canonical_solution']}

    Introduce a subtle bug in this function. Provide your response in JSON format with two fields:
    1. 'bug_explanation': A brief description of the bug you introduced.
    2. 'buggy_code': The function with the bug introduced (exclude the function signature and docstring).

    Ensure the bug is not too obvious and the code still looks plausible.
    """
    response = litellm.completion(
        model = LLM_NAME,
        messages = [{
            'role': 'user',
            'content': prompt
        }],
        response_format = {
            'type': 'json_schema',
            'json_schema': BugResponse.model_json_schema(),
            'strict': True
        },
        max_tokens = 500,
        temperature = 1.2,
        top_p = 0.05,
    )

    return json.loads(response.choices[0].message.content)


def create_buggy_problem(problem, max_attempts=3):
    for _ in range(max_attempts):
        try:
            bug_info = introduce_bug(problem)
            buggy_completion = bug_info['buggy_code']
            
            # Check if the buggy solution fails the tests
            result = check_correctness(problem, {"completion": buggy_completion})
            
            if not result['passed']:
                # Bug successfully introduced
                problem_with_bug = problem.copy()
                problem_with_bug['starting_code'] = buggy_completion
                problem_with_bug['bug_explanation'] = bug_info['bug_explanation']
                return problem_with_bug
        except Exception as e:
            print(f"Error in create_buggy_problem: {e}")
    
    # If we couldn't introduce a bug after max_attempts, return None
    return None


def process_problem(item):
    task_id, problem = item
    buggy_problem = create_buggy_problem(problem)
    return task_id, buggy_problem


def create_buggy_problems_parallel(problems, max_workers=10):
    buggy_problems = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(process_problem, item): item[0] for item in problems.items()}
        for future in concurrent.futures.as_completed(future_to_task):
            task_id = future_to_task[future]
            try:
                task_id, result = future.result()
                if result:
                    buggy_problems[task_id] = result
                else:
                    print(f"Failed to create buggy problem for task {task_id}")
            except Exception as e:
                print(f"Error processing task {task_id}: {e}")
    
    return buggy_problems

# Main execution
if __name__ == '__main__':
    problems = read_problems()
    buggy_problems = create_buggy_problems_parallel(problems)
    
    # Print some statistics
    print(f"Total problems: {len(problems)}")
    print(f"Buggy problems created: {len(buggy_problems)}")

    # Optionally, you can save the buggy_problems to a file
    with open('buggy_problems.json', 'w') as f:
        json.dump(buggy_problems, f, indent=2)
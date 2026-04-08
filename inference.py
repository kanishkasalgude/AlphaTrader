import os
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)


def _bool(v: bool) -> str:
    return "true" if v else "false"


def main() -> None:
    task = os.getenv("TASK_NAME", "click-test")
    env = os.getenv("ENV_NAME", "miniwob")

    rewards: list[str] = []
    success = False
    steps = 0

    print("[START] task={} env={} model={}".format(task, env, MODEL_NAME))

    try:
        _ = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
        )

        steps = 1
        reward = 0.00
        done = True
        error = None

        rewards.append(f"{reward:.2f}")
        print(
            "[STEP] step={} action={} reward={} done={} error={}".format(
                steps,
                "noop()",
                f"{reward:.2f}",
                _bool(done),
                "null" if error is None else str(error),
            )
        )

        success = True
    except Exception as e:
        steps = max(steps, 1)
        rewards.append(f"{0.00:.2f}")
        print(
            "[STEP] step={} action={} reward={} done={} error={}".format(
                steps,
                "error()",
                f"{0.00:.2f}",
                _bool(True),
                type(e).__name__,
            )
        )
        success = False
    finally:
        print(f"[END] success={_bool(success)} steps={steps} rewards={','.join(rewards)}")


if __name__ == "__main__":
    main()
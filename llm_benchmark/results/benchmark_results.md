## LLM Benchmark Results

| Model                   | Strategy   | Accuracy   |   Macro F1 |   F1 (Personality disorder) | Avg Latency   | Cost / 1k   |
|:------------------------|:-----------|:-----------|-----------:|----------------------------:|:--------------|:------------|
| MentalBERT (fine-tuned) | —          | 84.2%      |     0.838  |                      0.83   | ~local        | ~$0.00      |
| Claude Haiku            | zeroshot   | 59.8%      |     0.5763 |                      0.2632 | 0.94s         | $0.182      |
| Claude Haiku            | fewshot    | 65.3%      |     0.6341 |                      0.4091 | 0.94s         | $0.356      |
| Claude Haiku            | cot        | 41.0%      |     0.3414 |                      0      | 3.2s          | $1.014      |
| GPT-4o-mini             | zeroshot   | 61.8%      |     0.5998 |                      0.381  | 1.49s         | $0.033      |
| GPT-4o-mini             | fewshot    | 63.3%      |     0.6104 |                      0.375  | 0.72s         | $0.066      |
| GPT-4o-mini             | cot        | 59.4%      |     0.5794 |                      0.2927 | 3.27s         | $0.160      |

### Per-Class F1

| Model        | Strategy   |   Normal |   Depression |   Suicidal |   Anxiety |   Bipolar |   Stress |   Personality disorder |
|:-------------|:-----------|---------:|-------------:|-----------:|----------:|----------:|---------:|-----------------------:|
| MentalBERT   | —          |   0.9559 |       0.7848 |     0.7515 |    0.8715 |    0.887  |   0.785  |                 0.83   |
| Claude Haiku | zeroshot   |   0.7179 |       0.5263 |     0.6747 |    0.5981 |    0.8276 |   0.4262 |                 0.2632 |
| Claude Haiku | fewshot    |   0.8    |       0.5977 |     0.6923 |    0.6465 |    0.8621 |   0.4314 |                 0.4091 |
| Claude Haiku | cot        |   0.7609 |       0.3855 |     0.4516 |    0.4027 |    0.2222 |   0.1667 |                 0      |
| GPT-4o-mini  | zeroshot   |   0.7727 |       0.5476 |     0.6914 |    0.5714 |    0.8421 |   0.3922 |                 0.381  |
| GPT-4o-mini  | fewshot    |   0.76   |       0.5843 |     0.6933 |    0.6024 |    0.8667 |   0.3913 |                 0.375  |
| GPT-4o-mini  | cot        |   0.7945 |       0.5055 |     0.6835 |    0.5743 |    0.7719 |   0.4333 |                 0.2927 |

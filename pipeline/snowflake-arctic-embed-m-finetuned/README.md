---
base_model: Snowflake/snowflake-arctic-embed-m
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
- pearson_manhattan
- spearman_manhattan
- pearson_euclidean
- spearman_euclidean
- pearson_dot
- spearman_dot
- pearson_max
- spearman_max
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:40
- loss:CosineSimilarityLoss
widget:
- source_sentence: What role does NIST play in establishing AI standards?
  sentences:
  - "provides examples and concrete steps for communities, industry, governments,\
    \ and others to take in order to \nbuild these protections into policy, practice,\
    \ or the technological design process. \nTaken together, the technical protections\
    \ and practices laid out in the Blueprint for an AI Bill of Rights can help \n\
    guard the American public against many of the potential and actual harms identified\
    \ by researchers, technolo­"
  - "provides examples and concrete steps for communities, industry, governments,\
    \ and others to take in order to \nbuild these protections into policy, practice,\
    \ or the technological design process. \nTaken together, the technical protections\
    \ and practices laid out in the Blueprint for an AI Bill of Rights can help \n\
    guard the American public against many of the potential and actual harms identified\
    \ by researchers, technolo­"
  - "Acknowledgments: This report was accomplished with the many helpful comments\
    \ and contributions \nfrom the community, including the NIST Generative AI Public\
    \ Working Group, and NIST staﬀ and guest \nresearchers: Chloe Autio, Jesse Dunietz,\
    \ Patrick Hall, Shomik Jain, Kamie Roberts, Reva Schwartz, Martin \nStanley, and\
    \ Elham Tabassi. \nNIST Technical Series Policies \nCopyright, Use, and Licensing\
    \ Statements \nNIST Technical Series Publication Identifier Syntax \nPublication\
    \ History"
- source_sentence: What are the implications of AI in decision-making processes?
  sentences:
  - "The measures taken to realize the vision set forward in this framework should\
    \ be proportionate \nwith the extent and nature of the harm, or risk of harm,\
    \ to people's rights, opportunities, and \naccess. \nRELATIONSHIP TO EXISTING\
    \ LAW AND POLICY\nThe Blueprint for an AI Bill of Rights is an exercise in envisioning\
    \ a future where the American public is \nprotected from the potential harms,\
    \ and can fully enjoy the benefits, of automated systems. It describes princi­"
  - "state of the science of AI measurement and safety today. This document focuses\
    \ on risks for which there \nis an existing empirical evidence base at the time\
    \ this proﬁle was written; for example, speculative risks \nthat may potentially\
    \ arise in more advanced, future GAI systems are not considered. Future updates\
    \ may \nincorporate additional risks or provide further details on the risks identiﬁed\
    \ below."
  - "development of automated systems that adhere to and advance their safety, security\
    \ and \neffectiveness. Multiple NSF programs support research that directly addresses\
    \ many of these principles: \nthe National AI Research Institutes23 support research\
    \ on all aspects of safe, trustworthy, fair, and explainable \nAI algorithms and\
    \ systems; the Cyber Physical Systems24 program supports research on developing\
    \ safe"
- source_sentence: How are AI systems validated for safety and fairness according
    to NIST standards?
  sentences:
  - "tion and advises on implementation of the DOE AI Strategy and addresses issues\
    \ and/or escalations on the \nethical use and development of AI systems.20 The\
    \ Department of Defense has adopted Artificial Intelligence \nEthical Principles,\
    \ and tenets for Responsible Artificial Intelligence specifically tailored to\
    \ its national \nsecurity and defense activities.21 Similarly, the U.S. Intelligence\
    \ Community (IC) has developed the Principles"
  - "GOVERN 1.1: Legal and regulatory requirements involving AI are understood, managed,\
    \ and documented.  \nAction ID \nSuggested Action \nGAI Risks \nGV-1.1-001 Align\
    \ GAI development and use with applicable laws and regulations, including \nthose\
    \ related to data privacy, copyright and intellectual property law. \nData Privacy;\
    \ Harmful Bias and \nHomogenization; Intellectual \nProperty \nAI Actor Tasks:\
    \ Governance and Oversight"
  - "more than a decade, is also helping to fulﬁll the 2023 Executive Order on Safe,\
    \ Secure, and Trustworthy \nAI. NIST established the U.S. AI Safety Institute\
    \ and the companion AI Safety Institute Consortium to \ncontinue the eﬀorts set\
    \ in motion by the E.O. to build the science necessary for safe, secure, and \n\
    trustworthy development and use of AI. \nAcknowledgments: This report was accomplished\
    \ with the many helpful comments and contributions"
- source_sentence: How does the AI Bill of Rights protect individual privacy?
  sentences:
  - "match the statistical properties of real-world data without disclosing personally\
    \ \nidentiﬁable information or contributing to homogenization. \nData Privacy;\
    \ Intellectual Property; \nInformation Integrity; \nConfabulation; Harmful Bias\
    \ and \nHomogenization \nAI Actor Tasks: AI Deployment, AI Impact Assessment,\
    \ Governance and Oversight, Operation and Monitoring \n \nMANAGE 2.3: Procedures\
    \ are followed to respond to and recover from a previously unknown risk when it\
    \ is identiﬁed. \nAction ID"
  - "the principles described in the Blueprint for an AI Bill of Rights may be necessary\
    \ to comply with existing law, \nconform to the practicalities of a specific use\
    \ case, or balance competing public interests. In particular, law \nenforcement,\
    \ and other regulatory contexts may require government actors to protect civil\
    \ rights, civil liberties, \nand privacy in a manner consistent with, but using\
    \ alternate mechanisms to, the specific principles discussed in"
  - "civil rights, civil liberties, and privacy. The Blueprint for an AI Bill of Rights\
    \ includes this Foreword, the five \nprinciples, notes on Applying the The Blueprint\
    \ for an AI Bill of Rights, and a Technical Companion that gives \nconcrete steps\
    \ that can be taken by many kinds of organizations—from governments at all levels\
    \ to companies of \nall sizes—to uphold these values. Experts from across the\
    \ private sector, governments, and international"
- source_sentence: How does the AI Bill of Rights protect individual privacy?
  sentences:
  - "57 \nNational Institute of Standards and Technology (2023) AI Risk Management\
    \ Framework, Appendix B: \nHow AI Risks Diﬀer from Traditional Software Risks.\
    \ \nhttps://airc.nist.gov/AI_RMF_Knowledge_Base/AI_RMF/Appendices/Appendix_B \n\
    National Institute of Standards and Technology (2023) AI RMF Playbook. \nhttps://airc.nist.gov/AI_RMF_Knowledge_Base/Playbook\
    \ \nNational Institue of Standards and Technology (2023) Framing Risk"
  - "principles for managing information about individuals have been incorporated\
    \ into data privacy laws and \npolicies across the globe.5 The Blueprint for an\
    \ AI Bill of Rights embraces elements of the FIPPs that are \nparticularly relevant\
    \ to automated systems, without articulating a specific set of FIPPs or scoping\
    \ \napplicability or the interests served to a single particular domain, like\
    \ privacy, civil rights and civil liberties,"
  - "harmful \nuses. \nThe \nNIST \nframework \nwill \nconsider \nand \nencompass\
    \ \nprinciples \nsuch \nas \ntransparency, accountability, and fairness during\
    \ pre-design, design and development, deployment, use, \nand testing and evaluation\
    \ of AI technologies and systems. It is expected to be released in the winter\
    \ of 2022-23. \n21"
model-index:
- name: SentenceTransformer based on Snowflake/snowflake-arctic-embed-m
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: val
      type: val
    metrics:
    - type: pearson_cosine
      value: 0.6585006489314952
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.7
      name: Spearman Cosine
    - type: pearson_manhattan
      value: 0.582665729755017
      name: Pearson Manhattan
    - type: spearman_manhattan
      value: 0.6
      name: Spearman Manhattan
    - type: pearson_euclidean
      value: 0.6722783219807118
      name: Pearson Euclidean
    - type: spearman_euclidean
      value: 0.7
      name: Spearman Euclidean
    - type: pearson_dot
      value: 0.6585002582595083
      name: Pearson Dot
    - type: spearman_dot
      value: 0.7
      name: Spearman Dot
    - type: pearson_max
      value: 0.6722783219807118
      name: Pearson Max
    - type: spearman_max
      value: 0.7
      name: Spearman Max
---

# SentenceTransformer based on Snowflake/snowflake-arctic-embed-m

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [Snowflake/snowflake-arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [Snowflake/snowflake-arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m) <!-- at revision e2b128b9fa60c82b4585512b33e1544224ffff42 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'How does the AI Bill of Rights protect individual privacy?',
    'principles for managing information about individuals have been incorporated into data privacy laws and \npolicies across the globe.5 The Blueprint for an AI Bill of Rights embraces elements of the FIPPs that are \nparticularly relevant to automated systems, without articulating a specific set of FIPPs or scoping \napplicability or the interests served to a single particular domain, like privacy, civil rights and civil liberties,',
    'harmful \nuses. \nThe \nNIST \nframework \nwill \nconsider \nand \nencompass \nprinciples \nsuch \nas \ntransparency, accountability, and fairness during pre-design, design and development, deployment, use, \nand testing and evaluation of AI technologies and systems. It is expected to be released in the winter of 2022-23. \n21',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity
* Dataset: `val`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric             | Value   |
|:-------------------|:--------|
| pearson_cosine     | 0.6585  |
| spearman_cosine    | 0.7     |
| pearson_manhattan  | 0.5827  |
| spearman_manhattan | 0.6     |
| pearson_euclidean  | 0.6723  |
| spearman_euclidean | 0.7     |
| pearson_dot        | 0.6585  |
| spearman_dot       | 0.7     |
| pearson_max        | 0.6723  |
| **spearman_max**   | **0.7** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 40 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 40 samples:
  |         | sentence_0                                                                         | sentence_1                                                                          | label                                                            |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:-----------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              | float                                                            |
  | details | <ul><li>min: 12 tokens</li><li>mean: 14.43 tokens</li><li>max: 18 tokens</li></ul> | <ul><li>min: 41 tokens</li><li>mean: 80.55 tokens</li><li>max: 117 tokens</li></ul> | <ul><li>min: 0.53</li><li>mean: 0.61</li><li>max: 0.76</li></ul> |
* Samples:
  | sentence_0                                                                                    | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                | label                           |
  |:----------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------|
  | <code>What should business leaders understand about AI risk management?</code>                | <code>57 <br>National Institute of Standards and Technology (2023) AI Risk Management Framework, Appendix B: <br>How AI Risks Diﬀer from Traditional Software Risks. <br>https://airc.nist.gov/AI_RMF_Knowledge_Base/AI_RMF/Appendices/Appendix_B <br>National Institute of Standards and Technology (2023) AI RMF Playbook. <br>https://airc.nist.gov/AI_RMF_Knowledge_Base/Playbook <br>National Institue of Standards and Technology (2023) Framing Risk</code>        | <code>0.5692041097520776</code> |
  | <code>What kind of data protection measures are required under current AI regulations?</code> | <code>GOVERN 1.1: Legal and regulatory requirements involving AI are understood, managed, and documented.  <br>Action ID <br>Suggested Action <br>GAI Risks <br>GV-1.1-001 Align GAI development and use with applicable laws and regulations, including <br>those related to data privacy, copyright and intellectual property law. <br>Data Privacy; Harmful Bias and <br>Homogenization; Intellectual <br>Property <br>AI Actor Tasks: Governance and Oversight</code> | <code>0.5830958798587019</code> |
  | <code>What are the implications of AI in decision-making processes?</code>                    | <code>state of the science of AI measurement and safety today. This document focuses on risks for which there <br>is an existing empirical evidence base at the time this proﬁle was written; for example, speculative risks <br>that may potentially arise in more advanced, future GAI systems are not considered. Future updates may <br>incorporate additional risks or provide further details on the risks identiﬁed below.</code>                                  | <code>0.5317174553776045</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `eval_use_gather_object`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch | Step | val_spearman_max |
|:-----:|:----:|:----------------:|
| 1.0   | 3    | 0.6              |
| 2.0   | 6    | 0.7              |
| 3.0   | 9    | 0.7              |


### Framework Versions
- Python: 3.11.9
- Sentence Transformers: 3.1.1
- Transformers: 4.44.2
- PyTorch: 2.2.2
- Accelerate: 0.34.2
- Datasets: 3.0.0
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->
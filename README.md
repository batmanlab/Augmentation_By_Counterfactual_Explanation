# Augmentation by Counterfactual Explanation - Fixing an Overconfident Classifier 

This is the GitHub repo for the WACV'23 paper: "Augmentation by Counterfactual Explanation - Fixing an Overconfident Classifier" (https://arxiv.org/abs/2210.12196)

-- by [Sumedha Singla](https://www.linkedin.com/in/sumedhasingla), [Nihal Murali](https://scholar.google.co.in/citations?user=LVcXV4oAAAAJ&hl=en), [Forough Arabshahi](https://forougha.github.io/), [Sofia Triantafyllou](https://gr.linkedin.com/in/sofia-triantafillou-3b2160115), [Kayhan Batmanghelich](https://www.batman-lab.com/)

To cite our work, please use the following bibtex entry:

```bibtex
@inproceedings{singla2023augmentation,
  title={Augmentation by Counterfactual Explanation-Fixing an Overconfident Classifier},
  author={Singla, Sumedha and Murali, Nihal and Arabshahi, Forough and Triantafyllou, Sofia and Batmanghelich, Kayhan},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={4720--4730},
  year={2023}
}
```

## Example Use-Cases:
**general syntax:**    python <code_path> --config <config_path>

(1) Train a DenseNet classifier on the AFHQ dataset:
```
python ./Train_Classifier_DenseNet.py --config ./Configs/Classifier/DenseNet_CelebA.yaml
```

(2) Train a StyleGANv2 explainer on the CelebA dataset: 
```
python ./Explainer_StyleGANv2/Train_Explainer_StyleGANv2.py --config ./Configs/Explainer/styleGAN_CelebA.yaml
```

Once the classifier and explainer are trained, refer to *./Misc/retrain_clf_via_ace.ipynb* jupyter notebook for re-training the classifier using the proposed ACE method.

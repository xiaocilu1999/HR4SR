import torch


class Metrics(object):
    def __init__(self, args, model):
        self.metric_ks = args.metric_ks
        self.theta = args.theta
        self.model = model

    def calculate_metrics(self, sequence, candidates, labels, noise=None):

        denoise_output, feature_output, save_noisy = self.model.sampler(sequence, noise)

        inference_output = self.theta * denoise_output + (1 - self.theta) * feature_output

        scores = inference_output[:, -1, :]  # 得到最后一个词的线性表示
        scores[:, 0] = -999.999
        scores[:, -1] = -999.999

        scores = scores.gather(1, candidates)  # 在所有项目上评分

        metrics = self.recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)

        return metrics, save_noisy

    @staticmethod
    def recalls_and_ndcgs_for_ks(scores, labels, ks):
        metrics = {}
        result = {}

        answer_count = labels.sum(1)
        labels_float = labels.float()
        rank = (-scores).argsort(dim=1)

        cut = rank

        for k in sorted(ks, reverse=True):  # 这里必须是倒序  要不然切不动
            cut = cut[:, :k]

            hits = labels_float.gather(1, cut)
            metrics["Recall@%d" % k] = (
                (
                        hits.sum(1)
                        / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())
                )
                .mean()
                .cpu()
                .item()
            )

            position = torch.arange(2, 2 + k)
            weights = 1 / torch.log2(position.float())
            dcg = (hits * weights.to(hits.device)).sum(1)
            idcg = torch.Tensor([weights[: min(int(n), k)].sum() for n in answer_count]).to(
                dcg.device
            )
            ndcg = (dcg / idcg).mean()
            metrics["NDCG@%d" % k] = ndcg.cpu().item()

        for k in sorted(ks, reverse=False):
            result["Recall@%d" % k] = metrics["Recall@%d" % k]
        for k in sorted(ks, reverse=False):
            result["NDCG@%d" % k] = metrics["NDCG@%d" % k]

        return result

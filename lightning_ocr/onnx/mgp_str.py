import numpy as np
from lightning_ocr.onnx.base import BASE_ONNX_INFER, softmax
from lightning_ocr.onnx.tokenizer import OnnxTokenizer


class MGP_STR_ONNX(BASE_ONNX_INFER):
    def load_tokenizer(self):
        self.char = OnnxTokenizer.load_from_file(
            f"{self.model_folder}/char/vocab.json",
            f"{self.model_folder}/char/special_tokens_map.json",
        )
        self.bpe = OnnxTokenizer.load_from_file(
            f"{self.model_folder}/bpe/vocab.json",
            f"{self.model_folder}/bpe/special_tokens_map.json",
        )
        self.wp = OnnxTokenizer.load_from_file(
            f"{self.model_folder}/wp/vocab.json",
            f"{self.model_folder}/wp/special_tokens_map.json",
        )

    def _decode_helper(self, pred_logits, format):
        if format == "char":
            decoder = self.char.decode
            eos_token = self.char.eos_token_id
            bos_token = self.char.bos_token_id
        elif format == "bpe":
            decoder = self.bpe.decode
            eos_token = self.bpe.eos_token_id
            bos_token = self.bpe.bos_token_id
        elif format == "wp":
            decoder = self.wp.decode
            eos_token = self.wp.eos_token_id
            bos_token = self.wp.bos_token_id
        else:
            raise ValueError(f"Format {format} is not supported.")

        dec_strs, conf_scores = [], []
        batch_size = pred_logits.shape[0]
        batch_max_length = pred_logits.shape[1]

        # topk - numpy variant
        k = 1
        preds_index = np.argsort(-pred_logits, axis=-1)[:, :, :k]
        preds_index = preds_index.reshape(-1, batch_max_length)
        preds_max_prob = softmax(pred_logits, axis=2).max(axis=2)

        for index in range(batch_size):
            pred_eos_index = np.where(preds_index[index] == eos_token)[0]
            if len(pred_eos_index) > 0:
                if pred_eos_index[0] == 0:
                    pred_eos_index = pred_eos_index[1:]

                pred_eos_index = pred_eos_index[0]
            else:
                pred_eos_index = batch_max_length - 1

            pred_index = preds_index[index][:pred_eos_index]
            pred_max_prob = preds_max_prob[index][:pred_eos_index]

            if pred_index[0] == bos_token:
                pred_index = pred_index[1:]
                pred_max_prob = pred_max_prob[1:]

            pred = decoder(pred_index)
            dec_strs.append(pred)
            conf_scores.append(pred_max_prob)
        return dec_strs, conf_scores

    def postprocess(self, outputs):
        char_preds, bpe_preds, wp_preds = outputs
        batch_size = char_preds.shape[0]

        char_strs, char_scores = self._decode_helper(char_preds, "char")
        bpe_strs, bpe_scores = self._decode_helper(bpe_preds, "bpe")
        wp_strs, wp_scores = self._decode_helper(wp_preds, "wp")

        final_strs = []
        final_scores = []
        for i in range(batch_size):
            scores = [char_scores[i], bpe_scores[i], wp_scores[i]]
            strs = [char_strs[i], bpe_strs[i], wp_strs[i]]
            max_score_index = scores.index(
                max(scores, key=lambda score: score.cumprod()[-1])
            )
            final_strs.append(strs[max_score_index])
            final_scores.append(scores[max_score_index])

        return final_strs, final_scores

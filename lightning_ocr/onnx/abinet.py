import numpy as np
from lightning_ocr.onnx.base import BASE_ONNX_INFER


class ABINetVisionONNX(BASE_ONNX_INFER):
    def _decode_helper(self, pred_logits):
        decoder = self.tokenizer.decode
        eos_token = self.tokenizer.eos_token_id
        bos_token = self.tokenizer.bos_token_id

        dec_strs, conf_scores = [], []
        batch_size = pred_logits.shape[0]
        batch_max_length = pred_logits.shape[1]

        preds_index = np.argmax(pred_logits, axis=2)
        preds_max_prob = pred_logits.max(axis=2)

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
        dec_strs, conf_scores = self._decode_helper(outputs[0])
        return dec_strs, conf_scores

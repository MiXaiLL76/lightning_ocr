import torch


def load_mmocr_state_dict(path):
    state_dict = torch.load(path, map_location=torch.device("cpu"), weights_only=False)[
        "state_dict"
    ]

    for key in list(state_dict):
        if key == "backbone.conv1.bias":
            state_dict.pop(key)
            continue

        if "data_preprocessor" in key:
            state_dict.pop(key)
            continue

        if key.startswith("backbone"):
            new_key = None
            if ("layer" in key) and (".conv1." in key):
                new_key = key.replace(".conv1.", ".conv1x1.")

            if ("layer" in key) and (".conv2." in key):
                new_key = key.replace(".conv2.", ".conv3x3.")

            if new_key is not None:
                state_dict[new_key] = state_dict.pop(key)
            continue

        if key.startswith("encoder"):
            new_key = None
            if "encoder.transformer" in key:
                if "attentions" in key:
                    if ".attn." in key:
                        idx = key.split(".")[2]

                        # TO: encoder.transformer.0.attentions.0.attn.in_proj_weight
                        # BE: encoder.encoder_layer.self_attn.in_proj_weight

                        # TO: encoder.transformer.1.attentions.0.attn.in_proj_weight
                        # BE: encoder.transformer.layers.0.self_attn.in_proj_weight

                        if idx == "0":
                            new_key = key.replace(
                                f"transformer.{idx}.attentions.0.attn",
                                "encoder_layer.self_attn",
                            )
                        else:
                            new_key = key.replace(
                                f"transformer.{idx}.attentions.0.attn",
                                f"transformer.layers.{int(idx)-1}.self_attn",
                            )

            if "transformer" in key:
                if ("ffns" in key) and ("layers" in key):
                    # TO: encoder.transformer.0.ffns.0.layers.0.0.weight
                    # BE: encoder.encoder_layer.linear1.weight

                    # TO: encoder.transformer.1.ffns.0.layers.0.0.weight
                    # BE: encoder.transformer.layers.0.linear1.weigh

                    # TO: encoder.transformer.0.ffns.0.layers.1.weight torch.Size([512, 2048])
                    # BE: encoder_layer.linear2.weight torch.Size([512, 2048])

                    if "0.0" in key:
                        idx = key.split(".")[2]

                        if idx == "0":
                            new_key = key.replace(
                                f"transformer.{idx}.ffns.0.layers.0.0",
                                "encoder_layer.linear1",
                            )
                        else:
                            new_key = key.replace(
                                f"transformer.{idx}.ffns.0.layers.0.0",
                                f"transformer.layers.{int(idx)-1}.linear1",
                            )
                    else:
                        idx = key.split(".")[2]
                        idx2 = key.split(".")[6]

                        if idx == "0":
                            new_key = key.replace(
                                f"transformer.{idx}.ffns.0.layers.{idx2}",
                                "encoder_layer.linear2",
                            )
                        else:
                            new_key = key.replace(
                                f"transformer.{idx}.ffns.0.layers.{idx2}",
                                f"transformer.layers.{int(idx)-1}.linear2",
                            )

                elif "norms" in key:
                    # encoder.transformer.0.norms.0.weight
                    # encoder.encoder_layer.norm1.weight
                    # encoder.transformer.layers.0.norm1.weight
                    idx = key.split(".")[2]
                    idx2 = key.split(".")[4]
                    if idx == "0":
                        new_key = key.replace(
                            f"transformer.{idx}.norms.{idx2}",
                            f"encoder_layer.norm{int(idx2)+1}",
                        )
                    else:
                        new_key = key.replace(
                            f"transformer.{idx}.norms.{idx2}",
                            f"transformer.layers.{int(idx)-1}.norm{int(idx2)+1}",
                        )
                    # print(new_key)

            if new_key is not None:
                state_dict[new_key] = state_dict.pop(key)
            continue

        if key.startswith("decoder"):
            if ".vision_decoder" in key:
                new_key = key.replace(".vision_decoder", "")
                state_dict[new_key] = state_dict.pop(key)
                key = str(new_key)

            new_key = None
            if "k_encoder" in key:
                # decoder.k_encoder.0.conv.weight torch.Size([64, 512, 3, 3])
                # decoder.k_encoder.0.bn.weight torch.Size([64])
                # decoder.k_encoder.0.bn.bias torch.Size([64])
                # decoder.k_encoder.0.bn.running_mean torch.Size([64])
                # decoder.k_encoder.0.bn.running_var torch.Size([64])
                # decoder.k_encoder.0.bn.num_batches_tracked torch.Size([])

                # k_encoder.0.0.weight torch.Size([64, 512, 3, 3])
                # k_encoder.0.0.bias torch.Size([64])
                # k_encoder.0.1.weight torch.Size([64])
                # k_encoder.0.1.bias torch.Size([64])
                # k_encoder.0.1.running_mean torch.Size([64])
                # k_encoder.0.1.running_var torch.Size([64])
                # k_encoder.0.1.num_batches_tracked torch.Size([])
                idx = key.split(".")[2]
                idx_type = key.split(".")[3]
                new_idx_type = 0 if idx_type == "conv" else 1

                new_key = key.replace(
                    f"k_encoder.{idx}.{idx_type}", f"k_encoder.{idx}.{new_idx_type}"
                )
                if f"k_encoder.{idx}.1.bias" in new_key:
                    state_dict[new_key.replace("1.bias", "0.bias")] = state_dict[key]

            elif "k_decoder" in key:
                # decoder.k_decoder.0.1.conv.weight torch.Size([64, 64, 3, 3])
                # decoder.k_decoder.0.1.bn.weight torch.Size([64])
                # decoder.k_decoder.0.1.bn.bias torch.Size([64])
                # decoder.k_decoder.0.1.bn.running_mean torch.Size([64])
                # decoder.k_decoder.0.1.bn.running_var torch.Size([64])
                # decoder.k_decoder.0.1.bn.num_batches_tracked torch.Size([])

                # k_decoder.0.1.0.weight torch.Size([64, 64, 3, 3])
                # k_decoder.0.1.1.weight torch.Size([64])
                # k_decoder.0.1.1.bias torch.Size([64])
                # k_decoder.0.1.1.running_mean torch.Size([64])
                # k_decoder.0.1.1.running_var torch.Size([64])
                # k_decoder.0.1.1.num_batches_tracked torch.Size([])
                idx = key.split(".")[2]
                idx_type = key.split(".")[4]
                new_idx_type = "1.0" if idx_type == "conv" else "1.1"
                new_key = key.replace(
                    f"k_decoder.{idx}.1.{idx_type}", f"k_decoder.{idx}.{new_idx_type}"
                )
                if f"k_decoder.{idx}.1.1.bias" in new_key:
                    state_dict[new_key.replace("1.bias", "0.bias")] = state_dict[key]

            if new_key is not None:
                state_dict[new_key] = state_dict.pop(key)
                continue
    return state_dict

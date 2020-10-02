import torch
import torch.nn as nn

def translate_sentence(sentence, src_lang, trg_lang, model, device, max_len=50):
    model.eval()

    tokens = ['<BOS>'] + sentence + ['<EOS>']

    src_indexes = [src_lang.token2id[token] if token in src_lang.token2id else src_lang.token2id['<UNK>'] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_lang.token2id['<BOS>']]
    trg_conf = [] #output confidence

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        soft = nn.Softmax(dim=2)
        output =  soft(output)

        if i==0:
            confidence = output[0,0,:].detach().cpu().numpy()

        #trg_conf.append(float(output[0,0,pred_token].detach().cpu().numpy()))

        if pred_token == trg_lang.token2id['<EOS>']:
            break


    trg_tokens = [trg_lang.id2token[i] for i in trg_indexes]
    #confidence = {x:y   for x,y in zip(trg_tokens,trg_conf)}

    return trg_tokens[1:], attention, confidence
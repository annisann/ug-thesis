import torch


def train(model, data, criterion, optimizer):
    model.train()
    running_loss = 0.

    for input in data:
        input_utterances, v_act, a_act, d_act = input[0], input[1], input[2], input[3]

        v_pred, a_pred, d_pred = model(input_utterances)

        loss_v = criterion(v_pred, v_act)
        loss_a = criterion(a_pred, a_act)
        loss_d = criterion(d_pred, d_act)
        loss = loss_v + loss_a + loss_d

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss/len(data)


def evaluate(model, data, criterion):
    model.eval()
    running_loss = 0.

    with torch.no_grad():
        for input in data:
            input_utterances, v_act, a_act, d_act = input[0], input[1], input[2], input[3]

            v_pred, a_pred, d_pred = model(input_utterances)

            loss_v = criterion(v_pred, v_act)
            loss_a = criterion(a_pred, a_act)
            loss_d = criterion(d_pred, d_act)
            loss = loss_v + loss_a + loss_d

            running_loss += loss.item()

    return running_loss/len(data)

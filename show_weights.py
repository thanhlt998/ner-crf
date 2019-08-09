import eli5
from app import load_model

if __name__ == '__main__':
    model = load_model('data/model/crf_model_no_pos_chunk.pkl')

    with open('weights.html', mode='w', encoding='utf8') as f:
        f.write(eli5.show_weights(model.model))
        f.close()

from evaluation_utils import evaluate

if __name__ == '__main__':
    output = "test_data/deen_output"
    ref_bpe = "test_data/deen_ref_bpe"
    # expected_bleu_score = 22.5855084573
    bpe_bleu_score = evaluate(
        ref_bpe, output, "bleu", "bpe")
    print(bpe_bleu_score)

import sacrebleu

# Example sentences for testing
references = [['jag', 'tänker', 'på', 'rätten', 'till', 'tolerans', 'rätten', 'till', 'respekt', 'något', 'som', 'är', 'en', 'rättighet', 'för', 'alla', 'minoriteter', 'framför'], ['har', 'rådet', 'för', 'avsikt', 'att', 'ta', 'upp', 'den', 'här', 'frågan', 'inom', 'ramen', 'för', 'föranslutningsstrategin', 'och', 'att', 'kräva']]
candidate = [['Det', 'är', 'också', 'att', 'på', 'olika'], ['Det', 'finns', 'också', 'också', 'att', 'en', 'och', 'för', 'en', 'och', 'och', 'och', 'och', 'och', 'och', 'och', 'och']]

# Convert the lists of words back into sentences
references = [' '.join(ref) for ref in references]
candidate = ' '.join(candidate)

bleu = sacrebleu.raw_corpus_bleu([candidate], [references])
print(bleu.score)



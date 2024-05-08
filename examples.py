from transformers import MarianTokenizer


tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-sv')

# Example to check BOS and EOS handling
sample_text = "This is a sample sentence for checking BOS and EOS."
tokenized_output = tokenizer.encode(sample_text, add_special_tokens=True)
decoded_tokens = tokenizer.convert_ids_to_tokens(tokenized_output)

print("Tokenized output:", decoded_tokens)
print("First token, should be BOS (<s>):", decoded_tokens[0])
print("Last token, should be EOS (</s>):", decoded_tokens[-1])


#INCLUDE BOS IF NECESARY IN THE FUTURE

if tokenizer.bos_token_id is None:
    tokenizer.add_special_tokens({'bos_token': '<s>'})
    #print("Added <s> with ID:", tokenizer.convert_tokens_to_ids('<s>'))





#This is one for shwoing the inputs and labels 
'''
                for i in range(len(input_ids)):
            print('Input:', input_ids[i])
            
            print('---')
            print('Label:', labels[i])
            
            print('---')
            print('Attention:',attention_mask[i])
        
'''
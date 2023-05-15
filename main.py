import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text,num_return_sequences,num_beams):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text, type(tgt_text)

num_beams = 20
num_return_sequences = 20
contexts = ["My old company was a mess. There was no communication, no direction, and no one seemed to know what they were doing.",
"The management at my old company was terrible. They were always micromanaging and never gave me any credit for my work.",
"The culture at my old company was toxic. Everyone was always gossiping and backstabbing each other.",
"The work environment at my old company was horrible. I was constantly stressed and overworked.",
"I was never given any opportunities for advancement at my old company. I felt like I was stuck in a dead-end job.",
"The company is not good because the salary for is low and they make me to work overtime very often", 
"if i don't have the offer of this job, I think I gonna buy your company and fire you",
"The company was a mess. There was no communication, no direction, and no one seemed to know what they were doing.",
"The management at my old company was terrible. They were always micromanaging and never gave me any credit for my work.",
"The culture at my old company was toxic. Everyone was always gossiping and backstabbing each other.",
"The work environment at my old company was horrible. I was constantly stressed and overworked.",
"I was never given any opportunities for advancement at my old company. I felt like I was stuck in a dead-end job.",
"The company was always changing its priorities. I never knew what I was working on from one day to the next.",
"The company was constantly making promises that it couldn't keep. I was always disappointed.",
"The company was never willing to invest in its employees. I felt like I was just a number.",
"The company was always looking for ways to cut costs. This often meant cutting corners and sacrificing quality.",
"The company was never honest with its employees. I was always kept in the dark about what was going on.",
"The company was always blaming its employees for its problems. I never felt like I was part of a team.",
"The company was always taking credit for its employees' successes. I never felt like I was being appreciated.",
"The company was always putting its own interests ahead of its employees' interests. I never felt like I was being treated fairly.",
"The company was always breaking the law. I was always worried about getting in trouble.",
"The company was always unethical. I was always uncomfortable with the things I was asked to do.",
"The company was always taking advantage of its employees. I felt like I was being exploited.",
"The company was always lying to its customers. I felt like I was being used.",
"The company was always damaging the environment. I felt like I was contributing to a problem.",
"The company was always hurting people. I felt like I was part of a machine that was doing harm.",
"The company was always making the world a worse place. I felt like I was contributing to the problem.",
"This job is fucking ridiculous.",
"I can't believe I have to work with these asholes.",
"This company is a piece of shit.",
"I'm so sick of this job.",
"I'm going to quit this job as soon as I can.",
"Perfectionism Is My Greatest Weakness",
"I’m Really Nervous",
"I will do whateever",
"what the hell ?",
"I don't know anything about it", 
]
decoded_list = []

for context in contexts:
  results, is_type = get_response(context,num_return_sequences,num_beams)
  for decoded_string in results:
    print(decoded_string)
    decoded_list.append(decoded_string)

# print(decoded_list)
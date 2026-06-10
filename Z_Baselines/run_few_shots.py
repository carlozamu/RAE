"""
ERA: Evolving Reasoning Agents
Baseline Execution Script (No Primer + Logging)
"""
import asyncio
import os
import sys
import time
import gc
import torch
from tqdm.asyncio import tqdm

# Get the absolute path of the directory one level up (RAE)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Append it to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# --- Internal Modules ---
from Fitness.fitness import Fitness
from Data.clutrr import CLUTTRManager
from Utils.LLM import LLM
from Utils.utilities import just_log, log_and_print

# AUTO-GENERATED PROMPT TEMPLATES

PROMPT_TEMPLATES_0 = {
    'min_complexity': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Heidi] and her mother [Nancy] went to volunteer at the homeless shelter last weekend. [Heidi] and her husband [Jeff] were expecting a baby. [Jeff] was hoping for a boy.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Jeff is Nancy's?<end_of_turn>
<start_of_turn>model
son-in-law<end_of_turn>
<start_of_turn>user
Story: [Kevin] was speaking with his father, [Wayne], and with his wife, [Linda].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Wayne is Linda's?<end_of_turn>
<start_of_turn>model
father-in-law<end_of_turn>
<start_of_turn>user
Story: [Nicolas] and his sister [June] love going to the movies. Sometime [Nicolas] invites his grandmother [Ashley]
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. June is Ashley's?<end_of_turn>
<start_of_turn>model
granddaughter<end_of_turn>
""",
    'max_extrapolation': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Steven] enjoys reading books to his grandson, [Dan]. [Patrice] took a picture with her son, [Dan], and sent it to her brother, [James]. [Mabel] took her sister, [Sharon], out to dinner for her birthday. [Kathleen] went to eat with her parents, [James] and [Ellen]. [Kathleen]'s sister, [Sharon], was n't able to join them. [James] went fishing with his daughters [Kathleen] and [Nadia]. [Nadia]'s sister [Mabel] went swimming instead.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Steven is Ellen's?<end_of_turn>
<start_of_turn>model
father<end_of_turn>
<start_of_turn>user
Story: [Teresita] and her father, [William], went to the marina. [William]'s daughter, [Monica], had purchased a boat, and they were eager to see it. [Mary] and her uncle [Steve] went to the pet shop. [Mary] saw a puppy that she loved, so [Steve] bought it for her. [Margaret], who is the sister of [Patrice], is a lovely girl. [James] and his sister, [Patrice], went to a sneak preview of the Avengers movie and loved it. [Mary] and her sister, [Monica], frequently bicker. However, they always make up quickly. [Kathleen] took her father, [James], out to dinner. [William] and [Margaret] have a happy marriage. [Ellen] spent a great day shopping with her daughter, [Kathleen]. [William] went to see his daughter [Teresita] for parents day at college.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Steve is Ellen's?<end_of_turn>
<start_of_turn>model
brother<end_of_turn>
<start_of_turn>user
Story: [Constance] went to the mall, because she wanted to look for a present for her daughter, [Beatrice]. [Steven] was looking forward to dinner with his granddaughter, [Nadia]. [Dan] has a aunt named [Constance], routinely he went to school with his aunt [Don] and his brother [Sidney] had a tie score. [Sidney] asked his sister, [Beatrice], to come up with a tie-breaker. [Charles] went to the game with his sister [Nadia]. [Orville] got his son, [James], a car for his birthday. [Charles] was thrilled his brother, [Orville], was able to make it to the party. [Cesar] took his dad [Steven] to the game. [Cesar]'s son [Dan] was going to be playing
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Don is James's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
""",
    'uniform_median': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Katherine] attended the graduation of her sister's son [Michael]. [David]'s grandmother, [Mary], was prepping for a marathon at age 70! [Mary] asked her son [Joseph] if his son [Michael] wanted anything in particular for his birthday. [Bonnie] took her sister, [Dorothy], out to dinner for her birthday. [Bonnie] was excited to visit her son, [David], at college.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Katherine is Dorothy's?<end_of_turn>
<start_of_turn>model
sister<end_of_turn>
<start_of_turn>user
Story: [Warren] and his brother [Arthur] went to the park to play basketball. [Bonnie] went to the store with her sister [Dorothy] [Seth] and [Arthur] went to help their mother [Dorothy] pick up her husband [Alvin]'s new golf clubs. [Seth]'s mother, [Dorothy], waited impatiently for him at the diner.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Bonnie is Warren's?<end_of_turn>
<start_of_turn>model
aunt<end_of_turn>
<start_of_turn>user
Story: [Erica] and her brother [Clifton] enlisted in the army [Ross] was thrilled his brother, [Michael], was able to make it to the party. [Ross] adored his brother, [Michael]. [Michael]'s brother, [Patrick], also adored him. [Erica] and her son [Alfredo] made hot dogs. [Alfredo]'s brother [Michael] ate one.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Patrick is Clifton's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
""",
    'bounding_box': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Theresa] is very proud of her son. His name is [Tony]. [Wayne] took his sister [Theresa] out to dinner for her birthday.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Wayne is Tony's?<end_of_turn>
<start_of_turn>model
uncle<end_of_turn>
<start_of_turn>user
Story: [Katherine] had a daughter named [Charlsie]. [Bonnie]'s son, [William], went to have lunch with her sister, [Katherine]. [Robert] and his wife [Katherine] went to see a movie. [Katherine]'s daughter, [Jon], came with them. [Charlsie] went to the store with her sister [Jon]
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. William is Robert's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
<start_of_turn>user
Story: [Stanley] and his sister [Rita] bought a painting for [Stanley]'s father [Steve]. [Steve]'s brother [James] said it was an ugly painting. [Rita] went to her aunt [Patrice]'s house for dinner. [Patrice] made meatloaf, and it was delicious. [Kathleen] bought her mother, [Ellen], a puppy for her birthday. [Sharon] was glad to see her father, [James], at her dance recital. [Kathleen], who is the sister of [Mabel], is a lovely girl. [Ellen] had picked her daughter [Sharon] out the cutest new dress to wear on her birthday. [Mabel] went to the store with her sister [Kathleen] [Ellen] had picked her daughter [Kathleen] out the cutest new dress to wear on her birthday.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Patrice is Ellen's?<end_of_turn>
<start_of_turn>model
sister<end_of_turn>
""",
    'inverted_recency': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Monica] took her sister, [Eula], out to dinner for her birthday. [Kathleen] went over to her uncle [Cesar]'s house for dinner. [Willie] likes to visit his sister. Her name is [Eula]. [Margaret] asked her daughter, [Eula], if she would like to go to a movie with her on Saturday night. [Cesar] was so happy for his brother, [Steve], getting a promotion. [Ellen] went to the mall, because she wanted to look for a present for her daughter, [Kathleen]. [Steve] and his sister [Ellen] went out for ice cream. [William]'s daughter [Eula] was playing on the swings in the park. [Monica], his other daughter, was napping in the stroller. [Steve] bought tickets for him to see the game with his nephew, [Willie].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Steve is Margaret's?<end_of_turn>
<start_of_turn>model
brother<end_of_turn>
<start_of_turn>user
Story: [Seth]'s brother [Warren] would not let him play the xbox game. [Mary] loves playing tag with her daughter's son. His name is [Warren]. [Dorothy] spent a great day shopping with her daughter, [Lilly]. [Lilly] and her brother [Seth] are having an argument over when her dad, [Alvin], will be back from the store. [Alvin] was having trouble with his divorce because he only got to see his daughter [Lilly] a few times a week
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Mary is Dorothy's?<end_of_turn>
<start_of_turn>model
mother<end_of_turn>
<start_of_turn>user
Story: [Nancy]'s father, [Clarence], took her to the bowling alley. [Nancy]'s daughter, [Louise], joined them because bowling was her favorite hobby.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Clarence is Louise's?<end_of_turn>
<start_of_turn>model
grandfather<end_of_turn>
""",
    'strided_generalist': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Wayne] and his wife [Nancy] will be at the play. [Nancy] had a heart-to-heart conversation with her daughter [Linda] about some things.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Linda is Wayne's?<end_of_turn>
<start_of_turn>model
daughter<end_of_turn>
<start_of_turn>user
Story: [Margaretta], a female, took her husband who is a male, [Angel], out for lunch. [Milton] took his daughter [Irene] to the baseball game. [Irene]'s brother [Pat] hates baseball and stayed home. [Margaretta] enjoys playing cards with her brother. His name is [Pat]. [Guadalupe] went to the beach with her husband [Milton].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Angel is Guadalupe's?<end_of_turn>
<start_of_turn>model
son-in-law<end_of_turn>
<start_of_turn>user
Story: [Alice] went to her brother [Donald]'s birthday party [Elizabeth] went to the mall, because she wanted to look for a present for her daughter, [Marge]. [Roger]'s grandmother, [Elizabeth], was eager to spend a weekend with all of her grandchildren. [Manuel] went out for a picnic with his sister [Melody] and his daughter [Mary]. [Melody]'s brother [Donald] was going to meet them later. [Roger] took his sister [Mary] out to lunch after learning that she got accepted into her first choice for university. [Melody] went shoe shopping with her sister [Marge].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Melody is Alice's?<end_of_turn>
<start_of_turn>model
sister<end_of_turn>
""",
    'dist_aligned_control': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [George] took a cab to the mall with his sister [Theresa]. [Theresa]'s sister [Nancy] drove herself there.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. George is Nancy's?<end_of_turn>
<start_of_turn>model
brother<end_of_turn>
<start_of_turn>user
Story: [Hugh] read a bedtime story to his daughter [Marie]. [Bobby]'s wife, [Vanetta], just went to graduate. [Marie] called [Vanetta], her sister, to see if she wanted to go out to eat.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Hugh is Bobby's?<end_of_turn>
<start_of_turn>model
father-in-law<end_of_turn>
<start_of_turn>user
Story: [Louis] is frustrated teaching his son how to drive. His son is [Michael]. [Pedro] went his brother to pick up his brother's daughter [Jennifer] from a birthday party. [Jennifer] and her brother [Michael] went to the movies together. [Pedro] 'daughter, [Rebecca], just made it into grad school.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Rebecca is Louis's?<end_of_turn>
<start_of_turn>model
niece<end_of_turn>
""",
    'boundary_pusher': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [James] took his wife [Frances] and grandson [Pete] to a bakery. [Pete]'s sister [Carrie] does n't like sweets and did n't go.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Carrie is Frances's?<end_of_turn>
<start_of_turn>model
granddaughter<end_of_turn>
<start_of_turn>user
Story: [Wayne] took his brother [Michael] to Las Vegas to go gambling. [Michael] likes to go fishing with his sister. Her name is [Jennifer]. [Beverly] likes taking her son [Wayne] to school. [Pedro] likes to take his niece, [Jennifer], for walks by the beach.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Pedro is Beverly's?<end_of_turn>
<start_of_turn>model
brother<end_of_turn>
<start_of_turn>user
Story: [Gilbert] likes to visit his sister. Her name is [Hattie]. [Adeline] planned a trip to the zoo for her brother, [Gilbert]. They had a great time. [Samuel] wrote a poem for his wife [Angela]. [Samuel] asked his brother [Donald] if he would come help him fix his car next weekend. [Angela] went to the mall, because she wanted to look for a present for her daughter, [Hattie].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Adeline is Donald's?<end_of_turn>
<start_of_turn>model
niece<end_of_turn>
""",
    'hard_tail_anchor': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Marge] went to her son [Edgar]'s House [Marge], who is the sister of [Maria], is a lovely girl. [Donald]'s daughter [Jodi] wanted to go get ice cream. [Donald]'s father, [Luis], took her to get some. [Luis] took his grandson [Edgar] to the zoo. [Maria]'s son [Jeffery] is turning nine today, so she is throwing him a party. [James] and his sister [Alice] made a really big kite. [James] 'son [Jeffery] was so excited to fly it.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Jodi is Alice's?<end_of_turn>
<start_of_turn>model
niece<end_of_turn>
<start_of_turn>user
Story: [Loren] was excited because today she was going to the zoo with her uncle [John]. [Thelma] went shoe shopping with her sister [Amanda]. [Bryan] invited his sister [Amanda] to dinner this Friday because it is her birthday. [Betty] wanted her sister, [Santa], to go the same college she attends. [Santa] chose to attend college with her brother, [Milton], instead. [Ronald] took his daughter [Betty] to cheer practice. [Loren] went shoe shopping with her sister [Betty]. [Bryan] took his favorite son [John] to a baseball game where he caught a foul ball as a souvenir. [Milton] took his sister, [Betty], to a baseball game.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Thelma is Ronald's?<end_of_turn>
<start_of_turn>model
niece<end_of_turn>
<start_of_turn>user
Story: [Cesar]'s brother, [Steve], went to get ice cream with his sister, [Margaret]. [Don] took his sister [Beatrice] out to lunch after learning that she got accepted into her first choice for university. [Don] played basketball with his brother [Stanley]. [Patrice] asked her daughter, [Nora], if she would like to go to a movie with her on Saturday night. [Steve] and his wife [Constance] threw a surprise party for their daughter [Beatrice]. [Steve]'s sister [Margaret] came. [Nora] was running late to meet her brother, [Eric]. [Cesar], [Eric]'s father, called and let him know.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Stanley is Patrice's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
""",
    'bimodal_catalyst': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Clarence] and his wife, [Ashley], went to a football game. [Clarence] had a wonderful time, but [Ashley] was bored. [June] and her grandmother [Ashley] like to quilt together on Sunday afternoons.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. June is Clarence's?<end_of_turn>
<start_of_turn>model
granddaughter<end_of_turn>
<start_of_turn>user
Story: [Amy] went with her daughter [Jennifer] to pick up [Jennifer]'s husband [Michael] at the airport. His flight arrived early.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Amy is Michael's?<end_of_turn>
<start_of_turn>model
mother-in-law<end_of_turn>
<start_of_turn>user
Story: [James] took his sister [Patrice] out to lunch after learning that she got accepted into her first choice for university. [William] took his sister [Ellen] out to lunch after learning that she got accepted into her first choice for university. [James] dotes on his daughter, [Sharon]. He does n't pay as much attention to [Mabel], [Sharon]'s sister, because she's older. [Patrice] made a pizza for her son [Darryl]. [Darryl]'s sister [Nora] tried a piece and liked it. [Mabel] asked her mother, [Ellen], what they were going to have for lunch. [Eric] invited his sister [Nora] to dinner this Friday because it is her birthday. [Kathleen] was glad to see her father, [James], at her dance recital. [James] walked his daughter [Kathleen] down the aisle on her wedding day.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. William is Eric's?<end_of_turn>
<start_of_turn>model
uncle<end_of_turn>
""",
}

PROMPT_TEMPLATES_1 = {
    'min_complexity': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Clarence] went hunting with his son, [Dale]. [Dale] loves cooking with his daughter. Her name is [Rosetta].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Clarence is Rosetta's?<end_of_turn>
<start_of_turn>model
grandfather<end_of_turn>
<start_of_turn>user
Story: [Belva] asked her father [Dale] if she could spend the night with a friend and he said okay. [Ashley] took her granddaughter [Belva] shopping.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Ashley is Dale's?<end_of_turn>
<start_of_turn>model
mother<end_of_turn>
<start_of_turn>user
Story: [Nicholas] has a son named [Michael] that looks exactly like him. [Clarence] is taking his son [Nicholas] on a camping trip for the weekend.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Clarence is Michael's?<end_of_turn>
<start_of_turn>model
grandfather<end_of_turn>
""",
    'max_extrapolation': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [James] is very much in love with his wife. Her name is [Ellen]. [Nora] took her brother [Darryl] and other brother [Dan] with her to eat pasta at a pasta buffet. [Cesar] took his son [Darryl] out for ice cream. [Cesar] took his sister, [Ellen], to a baseball game. [Eric]'s brother [Darryl] would not let him play the xbox game. [Elsie] took her granddaughter [Mabel] to the cinema. [Darryl] bought a steak dinner for himself and his brother [Dan]. [James] dotes on his daughter, [Kathleen]. He does n't pay as much attention to [Mabel], [Kathleen]'s sister, because she's older.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Elsie is Eric's?<end_of_turn>
<start_of_turn>model
grandmother<end_of_turn>
<start_of_turn>user
Story: [Rita] and [Don] left that evening to go bowling. [Steve] got his son, [Steven], a car for his birthday. [Ellen] loved her nephew, [Stanley]. [Ellen] went to her father [Steven] Birthday party [Dan] and his son [Cesar] went to look at cars. [Dan] ended up buying the Mustang. [Dan] likes to spend time with his aunt, [Constance], who was excellent at cooking chicken. [Steve]'s wife, [Constance], surprised him on his birthday. He could n't believe she pulled it off. [Constance] and her son [Don] went out to lunch together yesterday. [Rita] went to her brother [Stanley]'s birthday party [Constance] and her son, [Stanley], went to the park to fly a kite. They had fun doing it all day.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Stanley is Cesar's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
<start_of_turn>user
Story: [Dan] took his Aunt [Constance] out for her favorite meal [Eric] invited his father [Cesar] and his brother [Dan] to go on a weekend fishing trip. [Mary] decided to give her uncle, [Steve], a call on his birthday. [Stanley] likes to visit his sister. Her name is [Beatrice]. [William] took his daughter, [Mary], to a baseball game. [Sidney] is having a meal with his father [Steve]. [Constance] and [Steve] are happy with each other. [Steve] took his daughter [Beatrice] out for dinner. [Sidney] loves his uncle, [William].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Stanley is Cesar's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
""",
    'uniform_median': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Seth] and his grandmother [Mary] went to the science museum. They both had fun, and learned some things, too. [Arthur] enjoys going fishing with his brother. His name is [Warren]. [Seth] and his brother [Warren] always played pranks on each other [Warren] was disappointed that his father, [Alvin], would n't be at the play to see him perform. [Warren] took his son [Alvin] out to play gold later that night. [Warren] played chess with his brother [Arthur].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Mary is Warren's?<end_of_turn>
<start_of_turn>model
grandmother<end_of_turn>
<start_of_turn>user
Story: [Ross] went to his brother [Michael]'s Birthday party [Ronald]'s aunt [Erica] likes to drink a little bit too much wine at family gatherings. [Patrick] asked his brother [Ross] if he would come help him fix his car next weekend. [Erica] was mad at her son, [Michael]. She found he'd been stealing from her purse. [Robert] would n't let his son [James] go to the park by himself. [James]'s brother [Ronald] offered to go with him.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Patrick is Robert's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
<start_of_turn>user
Story: [Patrick]'s father, [Joseph], went bowling with his sister, [Katherine]. [Katherine] and her son [Ronald] went out to lunch together yesterday. [Alfredo] went to the Farmer's market with his mother [Erica] and his brother [Patrick]. [Ronald] went to the game with his sister [Charlsie].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Charlsie is Erica's?<end_of_turn>
<start_of_turn>model
niece<end_of_turn>
""",
    'bounding_box': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Douglas] and his son [George] are playing a father-son game of basketball. [John] is waiting to go to dinner with his dad [George].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Douglas is John's?<end_of_turn>
<start_of_turn>model
brother<end_of_turn>
<start_of_turn>user
Story: [Alfredo] likes to watch baseball with his brother. His name is [Ross]. [Katherine] enjoyed a homemade dinner with her son [Ronald] [Charlsie] went to her aunt [Erica]'s house for dinner. [Erica] made meatloaf, and it was delicious. [Ronald] took his sister, [Charlsie], to a baseball game. [Alfredo] and [Patrick] were both still very upset with their mother, [Erica].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Ross is Katherine's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
<start_of_turn>user
Story: [Constance] and her son [Sidney] went out to lunch together yesterday. [Sharon] went shoe shopping with her sister [Nadia]. [Sharon] and her sister, [Mabel], frequently bicker. However, they always make up quickly. [Nadia] went shoe shopping with her sister [Sharon]. [Kathleen] went to the store with her sister [Sharon] [Ellen] was on her way to pick up her daughter [Sharon] from the park. [Sharon] had spent the entire day there playing soccer with her sister [Kathleen]. [Sidney]'s aunt, [Patrice], took him fishing [Ellen] was glad to see her father, [Steven], at her dance recital. [Steven] drove his daughter [Constance] to soccer practice.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Patrice is Mabel's?<end_of_turn>
<start_of_turn>model
aunt<end_of_turn>
""",
    'inverted_recency': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Cesar] has a few children, [Steven], Bradley and Claire [Eula] and her sister, [Mary], frequently bicker. However, they always make up quickly. [Mary] was always arguing with her mother's husband [William]. [Constance], her brother [James], and her son, [Stanley], went out to get some tacos. [Stanley]'s sister, [Beatrice], did n't come, because she hates tacos. [Cesar] went to the game with his sister [Ellen]. [Margaret] dropped her daughter [Eula] off at the movies, before dropping food off at her brother [James]'s house. [Steven] loved his granddaughter, [Beatrice].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Ellen is William's?<end_of_turn>
<start_of_turn>model
sister<end_of_turn>
<start_of_turn>user
Story: [Bonnie]'s son, [William], had to stay home from school with a cold. [Kenneth] played chess with his brother [William]. [Erica], who is the sister of [Bonnie], is a lovely girl. [David] got his son, [Clifton], a car for his birthday. [Kenneth] likes to watch baseball with his brother. His name is [William]. [William] played chess with his brother [David].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Erica is Clifton's?<end_of_turn>
<start_of_turn>model
sister<end_of_turn>
<start_of_turn>user
Story: [Clarence] and his wife [Ashley] really wanted to have a child. Everything they did seemed to not work. They were losing hope. [Ashley] was so proud of her granddaughter, [Mary], graduating at the top of her class.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Mary is Clarence's?<end_of_turn>
<start_of_turn>model
granddaughter<end_of_turn>
""",
    'strided_generalist': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Nancy] has been waiting around all day for her son, [Kevin]. [Nancy] invited her father [Clarence] to dinner.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Kevin is Clarence's?<end_of_turn>
<start_of_turn>model
grandson<end_of_turn>
<start_of_turn>user
Story: [Henry] bought gifts for his lovely sister, [Thelma] [William] drove his daughter [Patricia] to soccer practice. [Antonia] had picked her daughter [Guadalupe] out the cutest new dress to wear on her birthday. [William] and his sister, [Guadalupe] love it when their grandmother visits because she spoils them. She is coming this week to watch them while their parents are out of town. [Henry] likes to visit his sister. Her name is [Patricia].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Thelma is Antonia's?<end_of_turn>
<start_of_turn>model
granddaughter<end_of_turn>
<start_of_turn>user
Story: [Jodi] and [Edgar] left that evening to go bowling. [Luis] took his daughter [Melody] out for dinner. [John] took his sister [Jodi] too the zoo but his sister [Bridget] could n't come because she had to work. [Luis] loved to take his granddaughter [Jodi] fishing. [Jodi] wished her father [Donald] would come with them. [Jodi] was dancing with her sister [Bridget], when see saw her father [Donald] leaving the party.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Melody is Edgar's?<end_of_turn>
<start_of_turn>model
aunt<end_of_turn>
""",
    'dist_aligned_control': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Jerry] bought a new refrigerator for his mother [Lillian]. [Janet] and her husband [Jerry] just got back from their honeymoon in Hawaii. They had a wonderful time there.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Lillian is Janet's?<end_of_turn>
<start_of_turn>model
mother-in-law<end_of_turn>
<start_of_turn>user
Story: [Vanetta] and her mother [David] went dress shopping on Saturday. [Frances]'s daughter, [David], just graduated high school. [Michael] and her sister [Vanetta] decided to take a trip to Mexico together.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Michael is Frances's?<end_of_turn>
<start_of_turn>model
grandson<end_of_turn>
<start_of_turn>user
Story: [Antonio] took his son [Kevin] for boxing lessons. He felt the boy needed to toughen up. [Kevin] went fishing with his grandfather [Harold]. They spent all day together. [Emma] took her father [Antonio] out to dinner for Father's Day. [Antonio] 'daughter, [Emma], just made it into grad school.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Harold is Antonio's?<end_of_turn>
<start_of_turn>model
father<end_of_turn>
""",
    'boundary_pusher': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Hugh] took his father, [James], to a baseball game downtown. They invited [Brandon] along, because [Hugh] is [Brandon]'s father. [David] took her son, [Brandon], to go get baked goods from the local bakery.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. James is David's?<end_of_turn>
<start_of_turn>model
father<end_of_turn>
<start_of_turn>user
Story: [Karen] and her sister [Jennifer] are visiting [Jennifer]'s Mom [Beverly], [Beverly] son [Samuel] is there. [Nettie] is taking a trip to see her sister [Karen].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Samuel is Nettie's?<end_of_turn>
<start_of_turn>model
brother<end_of_turn>
<start_of_turn>user
Story: [Thelma] wanted to visit an art museum, so she asked her father, [William] to take her. [Patricia] and her brother [Benito] went to see a movie. [Thelma] asked her mother, [Melanie], what they were going to have for lunch. [Donald] likes to visit his sister. Her name is [Melanie]. [William] and his son [Benito] went to the cinema together.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Donald is Patricia's?<end_of_turn>
<start_of_turn>model
uncle<end_of_turn>
""",
    'hard_tail_anchor': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Manuel] loves to fish with his brother, [Dallas] and his daughter [Earline]. [James] and his sister, [Melody] love it when their grandmother visits because she spoils them. She is coming this week to watch them while their parents are out of town. [Roger] is [Earline]'s older brother. He has never liked any of her boyfriends. [Maria] asked her sister [Melody] if she wanted to go to the movies with her. [Melody] said no, so [Maria] went with her brother [Dallas] instead. [Mary] took her brother [Roger] to the park, but left her sister [Kim] at home.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. James is Kim's?<end_of_turn>
<start_of_turn>model
uncle<end_of_turn>
<start_of_turn>user
Story: [Charles] watched the football game with his son [Ronald]. [Charles]'s daughter [Willie] was bored and played with her phone instead. [Donna] asked her daughter, [Helen], if she would like to go to a movie with her on Saturday night. [John] would n't let his son [Timothy] go to the park by himself. [Timothy]'s brother [Bryan] offered to go with him. [Kyle] went to see his daughter [Maryann] for parents day at college. [Helen] and her uncle [Ronald] went to the pet shop. [Helen] saw a puppy that she loved, so [Ronald] bought it for her. [Maryann] watched a golf tournament with her aunt [Donna]. [Willie] went to the beach with her husband [John].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Bryan is Kyle's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
<start_of_turn>user
Story: [Dan] loves visiting his grandmother [Elsie] in Florida. [Stanley] and his brother, [Don] run a turtle farm. [Patrice] was mad at her son, [Dan]. She found he'd been stealing from her purse. [Constance]'s mother [Elsie] was teaching her how to teach when [Constance]'s husband [Steve] arrived home. [James] took his favorite son [Steven] to a baseball game where he caught a foul ball as a souvenir. [James] and his sister [Patrice] have been best friends ever since childhood. [Sidney] likes to watch baseball with his brother. His name is [Don]. [Stanley] and his brother [Don] always played pranks on each other [Sidney] got his son, [Steve], a car for his birthday.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Don is Steven's?<end_of_turn>
<start_of_turn>model
grandson<end_of_turn>
""",
    'bimodal_catalyst': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Nicholas] is taking his son [Dennis] on a camping trip for the weekend. [Dennis] and his brother [Glen] went to the carnival. [Glen] wanted to ride the Ferris wheel, but [Dennis] insisted they go on the bumper cars first.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Nicholas is Glen's?<end_of_turn>
<start_of_turn>model
father<end_of_turn>
<start_of_turn>user
Story: [April] is looking for a good gift for her sister [Ann]. [Clarence] is a doting grandfather to his granddaughter [April]. He loves to take [April] for ice cream.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Ann is Clarence's?<end_of_turn>
<start_of_turn>model
granddaughter<end_of_turn>
<start_of_turn>user
Story: [Elsie] is visiting her daughter, [Margaret]. [Eula] want's her mother, [Margaret], to take her to the zoo. [Constance] stayed with her mother [Elsie] before the birth of her son [Don] [Steve]'s son, [Don], came first in a relay race at his school. [Steve]'s son, [Stanley], felt jealous of [Don]. [William] and his sister [Patrice] have been best friends ever since childhood. [Steve] took his sister [Patrice] too the zoo but his sister [Ellen] could n't come because she had to work. [Eula] vowed to never trust her father, [William] with her debit card again.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Ellen is Stanley's?<end_of_turn>
<start_of_turn>model
aunt<end_of_turn>
""",
}

PROMPT_TEMPLATES_2 = {
    'min_complexity': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Ashley] took a picture with her granddaughter [Heidi]. [Ashley] went with her husband [Clarence] to get a nice dinner for their anniversary.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Heidi is Clarence's?<end_of_turn>
<start_of_turn>model
granddaughter<end_of_turn>
<start_of_turn>user
Story: [Nicholas] is a good character in the society, He has daughter namely called [Jennifer], He does not any expect from others. Also he is sports man. But due to family situation unable to participate in competition, Finally his daughter after some years won in that competition. so he proud bu his daughter [Clarence] decided to take his son, [Nicholas], on a ski vacation.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Jennifer is Clarence's?<end_of_turn>
<start_of_turn>model
granddaughter<end_of_turn>
<start_of_turn>user
Story: [Jeff] took his grandmother [Ashley] to the dog track. [Nancy]'s daughter Paula went to law school. Paula's brother [Jeff] ended up working in a fast food restaurant.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Ashley is Nancy's?<end_of_turn>
<start_of_turn>model
mother<end_of_turn>
""",
    'max_extrapolation': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Sidney] has a few children, [Steve], Bradley and Claire [William] took his daughter [Monica] to the park that afternoon to play. [Constance], a female, took her husband who is a male, [Steve], out for lunch. [Eula] and her sister, [Monica], frequently bicker. However, they always make up quickly. [Sidney] and his sister [Beatrice] have been best friends ever since childhood. [Elsie] threw her twin daughters, [Margaret] and [Ellen] a birthday party. [Margaret]'s husband [William] helped plan it. [Beatrice] loves her father, [Steve]. She ca n't stand his mother [Elsie], though.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Eula is Constance's?<end_of_turn>
<start_of_turn>model
niece<end_of_turn>
<start_of_turn>user
Story: [Elsie] loves her son [James] very much. He loves her, too. [Ellen]'s sister, [Patrice], wanted to plan her a surprise party. She asked [Ellen]'s daughter, [Kathleen], to help with the invites. [Steve] took his son [Sidney] to play at the park, and [Sidney]'s paternal grandmother [Elsie] tagged along. [James] walked his daughter [Kathleen] down the aisle on her wedding day. [Dan] went to his brother [Eric]'s Birthday party [Eric] took his son [Cesar] out for pizza. [Nora] went to dinner with her daughter [Patrice]. They then went to the park afterwards. [Nora] 'father [Cesar] became enraged when she failed to complete her homework. [Nora] was grounded for a month and unable to spend time with friends.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Dan is Steve's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
<start_of_turn>user
Story: [James] took his sister [Patrice] out to lunch after learning that she got accepted into her first choice for university. [William] took his sister [Ellen] out to lunch after learning that she got accepted into her first choice for university. [James] dotes on his daughter, [Sharon]. He does n't pay as much attention to [Mabel], [Sharon]'s sister, because she's older. [Patrice] made a pizza for her son [Darryl]. [Darryl]'s sister [Nora] tried a piece and liked it. [Mabel] asked her mother, [Ellen], what they were going to have for lunch. [Eric] invited his sister [Nora] to dinner this Friday because it is her birthday. [Kathleen] was glad to see her father, [James], at her dance recital. [James] walked his daughter [Kathleen] down the aisle on her wedding day.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. William is Eric's?<end_of_turn>
<start_of_turn>model
uncle<end_of_turn>
""",
    'uniform_median': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Ross] likes to watch baseball with his brother. His name is [Michael]. [Joseph] is taking his wife [Erica] out to dinner for their date tonight. [Bonnie]'s husband, [Clifton], went skiing with his son, [Kenneth]. [Erica] and her son [Ross] went out to lunch together yesterday. [Michael]'s aunt [Bonnie] took him to the store to do some last minute party shopping.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Kenneth is Joseph's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
<start_of_turn>user
Story: [Alfredo] got his son, [Joseph], a car for his birthday. [Patrick] and his brother [Alfredo] always played pranks on each other [Dorothy], who is the sister of [Katherine], is a lovely girl. [Mary] and her daughter [Dorothy] went to see a movie yesterday and then got ice cream afterwards. [Erica]'s mother [Mary] was teaching her how to teach when [Erica]'s husband [Joseph] arrived home.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Patrick is Katherine's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
<start_of_turn>user
Story: [Katherine]'s son, [Ronald], went to have lunch with her sister, [Dorothy]. [Dorothy] took her sister, [Bonnie], out to dinner for her birthday. [Ronald] took his sister, [Charlsie], to a baseball game. [Jon], who is the sister of [Charlsie], is a lovely girl. [Jon] went to her brother [James]'s birthday party
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. James is Bonnie's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
""",
    'bounding_box': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Nancy] loved her husband [Wayne]. They had a daughter named [Lorraine]. They live in a nice house.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Lorraine is Wayne's?<end_of_turn>
<start_of_turn>model
daughter<end_of_turn>
<start_of_turn>user
Story: [Katherine] loves her son [Ronald] very much. He loves her, too. [James] likes to visit his sister. Her name is [Jon]. [Jon] enjoys playing cards with her brother. His name is [Ronald]. [Ross] asked his aunt, [Katherine], to read him a story. [Katherine] unfortunately was n't able to, since she did n't have her glasses, and she was almost blind without them. [Patrick] went to the Farmer's market with his mother [Erica] and his brother [Ross].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. James is Erica's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
<start_of_turn>user
Story: [Margaret] was happy to babysit her niece, [Nadia]. [Nadia] was a great child. [James] and his daughter [Nadia] took a day off school to go to the zoo. [Teresita] asked her mother [Elsie] if she could go outside and play with her friends. [Nora] decided to give her uncle, [James], a call on his birthday. [Elsie] was excited to visit her son, [Cesar], at college. [Margaret] spent a great day shopping with her daughter, [Teresita]. [Stanley] was so happy for his brother, [Sidney], getting a promotion. [Sidney]'s mother, [Constance], waited impatiently for him at the diner. [Patrice]'s sister [Constance] was trying on new make-up that [Patrice]'s daughter [Nora] bought for her as a gift.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Cesar is Stanley's?<end_of_turn>
<start_of_turn>model
uncle<end_of_turn>
""",
    'inverted_recency': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Steven] took his daughter, [Constance], to a baseball game. [James] went to see his daughter [Mabel] for parents day at college. [Steve] invited his sister [Margaret] to dinner this Friday because it is her birthday. [Mabel] wanted to visit her grandmother [Elsie], but [Elsie] was asleep. [Nora] decided to give her uncle, [Steve], a call on his birthday. [James] likes to watch baseball with his brother. His name is [Cesar]. [Steven] is very much in love with his wife. Her name is [Elsie]. [Cesar] took his wife, [Patrice] on a picnic. [Patrice]'s daughter, [Nora] stayed home. [James] likes to visit his sister. Her name is [Margaret].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Constance is James's?<end_of_turn>
<start_of_turn>model
sister<end_of_turn>
<start_of_turn>user
Story: [Katherine] had to work a double shift and needed someone to watch her son [Ronald]. Luckily his sister [Jon] was free. [James] loved going to the store with his mom [Katherine]. She always bought him snacks [Mary] took her granddaughter [Charlsie] to the cinema. [Robert]'s daughter [Charlsie] was playing on the swings in the park. [Jon], his other daughter, was napping in the stroller.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Mary is James's?<end_of_turn>
<start_of_turn>model
grandmother<end_of_turn>
<start_of_turn>user
Story: [Nicholas] bought a book for his son [Dennis]. [Nicholas] and his sister [Nancy] went to the ice skating rink. [Nicholas] fell down and [Nancy] laughed at him.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Nancy is Dennis's?<end_of_turn>
<start_of_turn>model
aunt<end_of_turn>
""",
    'strided_generalist': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Ashley] took her granddaughter [Lorraine] to the movies. [Lorraine] loved superhero films. [Jeff] and his sister [Lorraine] stood in line at the fair.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Jeff is Ashley's?<end_of_turn>
<start_of_turn>model
grandson<end_of_turn>
<start_of_turn>user
Story: [Milton] invited his sister [Melanie] to dinner this Friday because it is her birthday. [Victoria] took her brothers [William] and [Samuel] to a baseball game. [Victoria]'s son [Richard] went dancing instead. [Melanie] and her brother [Samuel] went to see a movie.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Richard is Milton's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
<start_of_turn>user
Story: [Manuel] loves to fish with his brother, [Dallas] and his daughter [Earline]. [James] and his sister, [Melody] love it when their grandmother visits because she spoils them. She is coming this week to watch them while their parents are out of town. [Roger] is [Earline]'s older brother. He has never liked any of her boyfriends. [Maria] asked her sister [Melody] if she wanted to go to the movies with her. [Melody] said no, so [Maria] went with her brother [Dallas] instead. [Mary] took her brother [Roger] to the park, but left her sister [Kim] at home.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. James is Kim's?<end_of_turn>
<start_of_turn>model
uncle<end_of_turn>
""",
    'dist_aligned_control': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Dennis] and his father [Nicholas] went to the golf course in town on Saturday and had a great time. [Clarence] decided to take his son, [Nicholas], on a ski vacation.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Dennis is Clarence's?<end_of_turn>
<start_of_turn>model
grandson<end_of_turn>
<start_of_turn>user
Story: [Hugh] 'wife [David] went with him to a cruise. [David]'s daughter [Vanetta] stayed home and studied. [Hugh] and his brother [Francisco] decided to play paintball on Saturday.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Vanetta is Francisco's?<end_of_turn>
<start_of_turn>model
niece<end_of_turn>
<start_of_turn>user
Story: [Marie] does n't like her brother [Louis]. [Karen] take care of her grandmother [Shantel] [Marie] could n't wait to pick up her husband [Robert] at the airport. [Karen] realized that her father [Louis] was gay.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Shantel is Robert's?<end_of_turn>
<start_of_turn>model
mother-in-law<end_of_turn>
""",
    'boundary_pusher': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Don]'s mother, [Lynn], took flowers to her sister, [David]. [David] would deliver them to her mother, [Frances].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Don is Frances's?<end_of_turn>
<start_of_turn>model
grandson<end_of_turn>
<start_of_turn>user
Story: [Harold] enjoyed rough housing with his son [Antonio]. [Harold]'s wife [Shantel] did n't approve of it though. [Shantel] went to visit her grandmother, [Michelle], in the retirement home. [Robert] took her daughter [Michelle] for ice cream.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Antonio is Robert's?<end_of_turn>
<start_of_turn>model
brother<end_of_turn>
<start_of_turn>user
Story: [Arlene] and her mother [Milton] made breakfast together. [Guadalupe] and her daughter [Margaretta] went to see a movie yesterday and then got ice cream afterwards. [Milton] drove his daughter [Margaretta] to soccer practice. [Irene] vowed to never trust her father, [Milton] with her debit card again. [Arlene]'s father, [Milton], felt a wave of sadness when [Arlene] left for college.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Guadalupe is Irene's?<end_of_turn>
<start_of_turn>model
mother<end_of_turn>
""",
    'hard_tail_anchor': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Jeffery] took his sister [Leslie] out to lunch after learning that she got accepted into her first choice for university. [Leslie] and [Richard] left that evening to go bowling. [Richard] took his sister [Leslie] out to lunch after learning that she got accepted into her first choice for university. [Richard] went to the Farmer's market with his mother [Maria] and his brother [Lucas]. [James] went fishing with his son [Jeffery]. [James]'s brother [Donald] wanted to go fishing with them, but he could n't make it. [Leslie] planned a trip to the zoo for her brother, [Lucas]. They had a great time.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Donald is Maria's?<end_of_turn>
<start_of_turn>model
brother<end_of_turn>
<start_of_turn>user
Story: [Willie] picked her son [Bryan] up from school and dropped him off at her husband, [John]'s office. [Marie] decided to give her uncle, [John], a call on his birthday. [Betty] planned a trip to the zoo for her brother, [Milton]. They had a great time. [Ronald] rushed to the hospital to find out that his wife and already given birth to a boy and had named him [Milton]. [Donna] had picked her daughter [Marie] out the cutest new dress to wear on her birthday. [Bryan]'s brother, [Timothy], went to get ice cream with his sister, [Amanda]. [Betty] watched a golf tournament with her aunt [Donna].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Amanda is Ronald's?<end_of_turn>
<start_of_turn>model
niece<end_of_turn>
<start_of_turn>user
Story: [Margaret] and her daughter [Monica] went to see a movie yesterday and then got ice cream afterwards. [William] went to the mall with his sisters [Constance] and [Margaret]. Later, [Margaret] went to the movies with her sister [Patrice]. [William] was so proud of his son, [Willie]. he received a great scholarship to college. [Eula] asked her mother, [Margaret], what they were going to have for lunch. [Margaret] took her daughter [Teresita] out shopping for new clothes. [Teresita]'s sister [Monica] came along. While she was there, [Monica] bought her brother [Willie] a new shirt. [Eula] went to dinner with her daughter [Margaret]. They then went to the park afterwards.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Patrice is Monica's?<end_of_turn>
<start_of_turn>model
aunt<end_of_turn>
""",
    'bimodal_catalyst': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Nicholas] takes his son [Dennis] to the movie theater. [Ashley], [Dennis]'s grandmother, waits for the two at home.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Ashley is Nicholas's?<end_of_turn>
<start_of_turn>model
mother<end_of_turn>
<start_of_turn>user
Story: [Heidi] and her mother [Nancy] went to volunteer at the homeless shelter last weekend. [Heidi] and her husband [Jeff] were expecting a baby. [Jeff] was hoping for a boy.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Jeff is Nancy's?<end_of_turn>
<start_of_turn>model
son-in-law<end_of_turn>
<start_of_turn>user
Story: [Steven] is buying his daughter [Patrice] a brand new car for her birthday. [James] went with his brothers [Steve] and [Cesar] to the new bar downtown. [Cesar]'s sister [Ellen], showed up with some friends later in the night. [Ellen] had picked her daughter [Sharon] out the cutest new dress to wear on her birthday. [Steven] would n't let his son [James] go to the park by himself. [James]'s brother [Steve] offered to go with him. [William] took his daughter [Eula] to the baseball game. [Eula]'s brother [Willie] hates baseball and stayed home. [Sharon] decided to give her uncle, [William], a call on his birthday.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Willie is Patrice's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
""",
}

PROMPT_TEMPLATES_3 = {
    'min_complexity': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Rosetta] stayed with his grandmother [Ashley] last summer on her farm. He had a great time. [Theresa] and her daughter [Rosetta] went to church. They were both very religious.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Ashley is Theresa's?<end_of_turn>
<start_of_turn>model
mother<end_of_turn>
<start_of_turn>user
Story: [June] and husband [James] disagree about whether to send their two children to summer school this year. [June]'s father [Dale] has volunteered to watch the kids as an alternative.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Dale is James's?<end_of_turn>
<start_of_turn>model
father-in-law<end_of_turn>
<start_of_turn>user
Story: [Ashley] took her daughter, [Amy], to lunch. [Amy] was honored to take her son [Michael] to the mother son dance at her son's school last Friday.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Ashley is Michael's?<end_of_turn>
<start_of_turn>model
grandmother<end_of_turn>
""",
    'max_extrapolation': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Cesar]'s brother, [Steve], went to get ice cream with his sister, [Margaret]. [Don] took his sister [Beatrice] out to lunch after learning that she got accepted into her first choice for university. [Don] played basketball with his brother [Stanley]. [Patrice] asked her daughter, [Nora], if she would like to go to a movie with her on Saturday night. [Steve] and his wife [Constance] threw a surprise party for their daughter [Beatrice]. [Steve]'s sister [Margaret] came. [Nora] was running late to meet her brother, [Eric]. [Cesar], [Eric]'s father, called and let him know.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Stanley is Patrice's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
<start_of_turn>user
Story: [Cesar] likes to visit his sister. Her name is [Constance]. [Steven] and his son, [Steve], went to the supermarket. [Cesar], [Steven]'s son, had asked them to bring him back some cookies. [Don] and his brother [Stanley] always played pranks on each other [Nora] watched a golf tournament with her aunt [Ellen]. [Patrice] was happy to babysit her niece, [Teresita]. [Teresita] was a great child. [Margaret] and her daughter [Teresita] like to look at the stars on clear nights. [Margaret] often borrows the telescope that belongs to her brother, [Steve], so they can see the stars better. [Patrice] had picked her daughter [Nora] out the cutest new dress to wear on her birthday. [Don] loved going to the store with his mom [Constance]. She always bought him snacks
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Ellen is Stanley's?<end_of_turn>
<start_of_turn>model
aunt<end_of_turn>
<start_of_turn>user
Story: [Kathleen] went to her aunt [Constance]'s house for dinner. [Constance] made meatloaf, and it was delicious. [James] and his brother [Cesar] played harmonicas together. [James]'s daughter [Kathleen] loved it. [Stanley] likes to spend time with his aunt, [Ellen], who was excellent at cooking chicken. [Eula]'s mother [Margaret] went to visit her mother [Elsie] and she took [Eula]'s brother [Willie] with her. [Constance] was excited to visit her son, [Stanley], at college. [Steve] and his brother, [Cesar] run a turtle farm. [Elsie] and her daughter [Ellen] went to see a movie yesterday and then got ice cream afterwards.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Willie is Steve's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
""",
    'uniform_median': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Dorothy] went to her father [Daniel] Birthday party [Alvin] took his son's [Warren] and [Seth] to school. [Dorothy] baked some cookies for her son [Seth] because it was his birthday. [Daniel] walked his daughter [Erica] down the aisle on her wedding day. [Warren] and his sister, [Lilly] love it when their grandmother visits because she spoils them. She is coming this week to watch them while their parents are out of town.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Erica is Lilly's?<end_of_turn>
<start_of_turn>model
aunt<end_of_turn>
<start_of_turn>user
Story: [Joseph] rushed to the hospital to find out that his wife and already given birth to a boy and had named him [Alfredo]. [Katherine] called her son, [Ronald], from upstairs so that he could eat lunch. [Ronald] got his sister, [Jon], so she could eat lunch as well. [Bonnie] wanted to buy her mother [Mary] flowers for Mothers Day, so she called her sister [Katherine] to go shopping with her. [Alfredo]'s grandmother, [Mary], was prepping for a marathon at age 70!
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Jon is Joseph's?<end_of_turn>
<start_of_turn>model
niece<end_of_turn>
<start_of_turn>user
Story: [Mary] had to work a double shift and needed someone to watch her son [Robert]. Luckily his sister [Erica] was free. [Erica]'s husband, [Joseph], went skiing with his son, [Ross]. [Ross] and his brother [Michael] went to the park to play basketball. [Mary] and her daughter [Bonnie] went to see a movie yesterday and then got ice cream afterwards.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Michael is Bonnie's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
""",
    'bounding_box': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [John] had always loved visits from his Aunt [Amy], as she was funny and had an outgoing personality. [George] is so proud of his son. [John] is the son's name.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Amy is George's?<end_of_turn>
<start_of_turn>model
sister<end_of_turn>
<start_of_turn>user
Story: [Clifton] was so proud of his son, [William]. he received a great scholarship to college. [Bonnie] baked chocolate muffins for her son [David] and daughter [Julie]. [Joseph]'s nephew [David] was happy to hear that he will be arriving at the airport later tonight. [Clifton] took his son [David] to the park to feed the squirrels. [David] is [Julie]'s older brother. He has never liked any of her boyfriends.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Joseph is William's?<end_of_turn>
<start_of_turn>model
uncle<end_of_turn>
<start_of_turn>user
Story: [Dan] took his Aunt [Constance] out for her favorite meal [Eric] invited his father [Cesar] and his brother [Dan] to go on a weekend fishing trip. [Mary] decided to give her uncle, [Steve], a call on his birthday. [Stanley] likes to visit his sister. Her name is [Beatrice]. [William] took his daughter, [Mary], to a baseball game. [Sidney] is having a meal with his father [Steve]. [Constance] and [Steve] are happy with each other. [Steve] took his daughter [Beatrice] out for dinner. [Sidney] loves his uncle, [William].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Stanley is Cesar's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
""",
    'inverted_recency': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Don] was playing tennis with his sister [Rita]. Afterwards, he was planning on playing chess with his sister [Beatrice]. [Nadia] was always arguing with her mother's husband [James]. [Rita] was always arguing with her mother's husband [Steve]. [Patrice] and her son [Eric] went out to lunch together yesterday. [Eric] loves talking to his grandfather [Steven] on the phone. [Stanley] took his mother [Constance] to the bakery. [Constance] wanted to buy her daughter [Beatrice] a cake. [Steven] 'son, [Steve], aced his math class and made his father proud. [Nadia] and her aunt, [Patrice], went to the deli. They got a half a pound of corned beef, and two pounds of salami.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Stanley is James's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
<start_of_turn>user
Story: [Erica] went to her son [Patrick]'s House [Patrick] got his son, [Joseph], a car for his birthday. [Joseph] and his son [Patrick] went to the cinema together. [Clifton] and his uncle [Ross] went to the county fair and went on the ferris wheel [Patrick]'s brother, [Ross], recently made it into town. [Erica] took her son [Michael] to school because he missed the bus.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Clifton is Michael's?<end_of_turn>
<start_of_turn>model
uncle<end_of_turn>
<start_of_turn>user
Story: [Rosetta] spent the whole month of January at her grandfather [Clarence]'s house. [Theresa] and her daughter [Rosetta] went to church. They were both very religious.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Clarence is Theresa's?<end_of_turn>
<start_of_turn>model
father<end_of_turn>
""",
    'strided_generalist': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Jacob] went to visit his sister [Linda] to the hospital last week. [Linda] took her father [Wayne] to watch a movie together.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Wayne is Jacob's?<end_of_turn>
<start_of_turn>model
father<end_of_turn>
<start_of_turn>user
Story: [Angela] loved her nephew, [Benito]. [Benito] likes to visit his sister. Her name is [Patricia]. [William] took his daughter [Patricia] and son [Benito] to the baseball game. [Benito] was sad that his brother [Henry] could n't go to the game because he was sick.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Angela is Henry's?<end_of_turn>
<start_of_turn>model
aunt<end_of_turn>
<start_of_turn>user
Story: [Dallas] is out playing tennis with his daughter [Shana] because it is nice and sunny outside. [Cecelia] wanted to buy her mother [Melody] flowers for Mothers Day, so she called her sister [Shana] to go shopping with her. [Cecelia] decided to give her uncle, [Donald], a call on his birthday. [John] and his sister, [Bridget] love it when their grandmother visits because she spoils them. She is coming this week to watch them while their parents are out of town. [Cecelia] took her sister, [Katherine], out to dinner for her birthday. [Dallas] took his daughter, [Katherine], to a baseball game. [Donald] went to see his daughter [Bridget] for parents day at college.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. John is Melody's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
""",
    'dist_aligned_control': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [George] and his daughter, [Janet] go to concerts every week. [Janet] got a big birthday cake from her aunt [Theresa].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Theresa is George's?<end_of_turn>
<start_of_turn>model
sister<end_of_turn>
<start_of_turn>user
Story: [Joshua] asked his mother [Frances] to read to him. [Frances] was too busy. [Frances] drove her son [Hugh] to his soccer game. [Hugh] bought his daughter, [Vanetta], flowers.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Vanetta is Joshua's?<end_of_turn>
<start_of_turn>model
niece<end_of_turn>
<start_of_turn>user
Story: [Shantel] loves cooking for her son. His name is [Pedro]. [Shantel] and [Harold] [Pedro] went to the store with his brother, [Louis]. [Louis] and his daughter [Jennifer] helped to make Thanksgiving dinner together.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Jennifer is Harold's?<end_of_turn>
<start_of_turn>model
granddaughter<end_of_turn>
""",
    'boundary_pusher': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Hugh] 'daughter, [Marie], just made it into grad school. [Cindy]'s brother [Joshua] took her to the fair when she was little. [Joshua] also brought his brother [Hugh].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Marie is Cindy's?<end_of_turn>
<start_of_turn>model
niece<end_of_turn>
<start_of_turn>user
Story: [Elsie] is very much in love with her husband. His name is [Ronald]. [Allan] and his brother [John] like to go to the movies every weekend. [Elsie] has started to accompany her brother [John] to the movies whenever he goes. [Allan]'s mother, [Marie], just got her citizenship.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Marie is Ronald's?<end_of_turn>
<start_of_turn>model
mother-in-law<end_of_turn>
<start_of_turn>user
Story: [Irene] and her Grandfather [Jose] enjoyed feeding the ducks at the park [Irene] went shoe shopping with her sister [Margaretta]. [Arlene] 'father [Milton] became enraged when she failed to complete her homework. [Arlene] was grounded for a month and unable to spend time with friends. [Milton] took his daughter, [Arlene], to a baseball game. [Margaretta] went shoe shopping with her sister [Arlene].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Jose is Arlene's?<end_of_turn>
<start_of_turn>model
grandfather<end_of_turn>
""",
    'hard_tail_anchor': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Marge] went to dinner with her daughter [Elizabeth]. They then went to the park afterwards. [James] and his children [Lucas] and [Leslie] visited his mother [Elizabeth] in the nursing home. [Jeffery] enjoys going fishing with his brother. His name is [Lucas]. [Richard] enjoys going fishing with his brother. His name is [Jeffery]. [Luis] wrote a poem for his wife [Elizabeth]. [Elizabeth] had a daughter named [Marge].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Richard is Luis's?<end_of_turn>
<start_of_turn>model
grandson<end_of_turn>
<start_of_turn>user
Story: [Donna] baked a cake for her son [Wallace]. For her daughter, [Marie], [Donna] made some fudge brownies. [Wallace] went to lunch with his aunt [Bonita]. [Willie] asked her brother [Kyle] if [Kyle]'s son [Frank] was sick. [Marie] went shoe shopping with her sister [Leah]. [Mark] and his uncle [Frank] went to the county fair and went on the ferris wheel [Helen] was eager to show her new house to her father, [Mark]. It was even better when her sister, [Leah], showed up, too.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Bonita is Willie's?<end_of_turn>
<start_of_turn>model
sister<end_of_turn>
<start_of_turn>user
Story: [Eula] took her sister, [Teresita], out to dinner for her birthday. [Teresita] and her sister, [Monica], frequently bicker. However, they always make up quickly. [Mary] went to the store with her sister [Teresita] [Mary]'s grandfather, [Steven], accompanied her to the basketball match [Teresita] went to the store with her sister [Eula] [Cesar] and [Ellen] asked their mother, if they could go play in the pool. [Cesar] bought to dress for his father [Steven] [Margaret] spent a great day shopping with her daughter, [Monica]. [Monica] went shoe shopping with her sister [Eula]. [Monica] asked her mother, [Margaret], what they were going to have for lunch.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Ellen is Eula's?<end_of_turn>
<start_of_turn>model
aunt<end_of_turn>
""",
    'bimodal_catalyst': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Nancy] took her younger sister [Lillian] to the carnival last weekend. [Andrea]'s mom, [Lillian], has recently taken art classes.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Nancy is Andrea's?<end_of_turn>
<start_of_turn>model
aunt<end_of_turn>
<start_of_turn>user
Story: [Milton] hated [Nicholas], so [Milton] ran away from home. [Milton] figured [Nicholas] did n't like being hid dad anyhow. [Theresa] is a sister of [Nicholas].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Theresa is Milton's?<end_of_turn>
<start_of_turn>model
aunt<end_of_turn>
<start_of_turn>user
Story: [Eric] took his sister, [Nora], to a baseball game. [Constance] took her brother [Cesar] to the park, but left her sister [Ellen] at home. [Dan] loved going to the store with his mom [Patrice]. She always bought him snacks [Patrice] and her husband [Cesar] wanted a child badly [Eric] took his son [Cesar] out to play gold later that night. [Dan] and his son [Cesar] went to look at cars. [Dan] ended up buying the Mustang. [Patrice] took her kids [Darryl], [Nora], and [Eric] to the store.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Darryl is Ellen's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
""",
}

PROMPT_TEMPLATES_4 = {
    'min_complexity': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Ashley] baked cookies to send to her grandson [Michael]. [Michael] split them with his sister [Emily] after they arrived.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Ashley is Emily's?<end_of_turn>
<start_of_turn>model
grandmother<end_of_turn>
<start_of_turn>user
Story: [Clarence] is a benevolent grandfather, and [Dennis] is his young grandson. [Dennis] asked his son, [Nicholas], to help paint his house.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Clarence is Nicholas's?<end_of_turn>
<start_of_turn>model
father<end_of_turn>
<start_of_turn>user
Story: [Emily]'s mom, [Amy], has recently taken art classes. [Dennis] took the kids to a babysitter so he could have a great date with his wife, [Emily].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Amy is Dennis's?<end_of_turn>
<start_of_turn>model
mother-in-law<end_of_turn>
""",
    'max_extrapolation': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Charles]'s mother, [Ellen], took him shopping for a new television [Ellen] and her sister, [Margaret], frequently bicker. However, they always make up quickly. [Charles] is [Nadia]'s older brother. He has never liked any of her boyfriends. [Mabel]'s mother, [Ellen], wanted to throw a surprise birthday party for [Ellen]'s sister, [Margaret], this weekend. [Nadia] just had a baby and presented the baby proudly to the new maternal grandmother, [Ellen]. [Cesar]'s mother, [Elsie], waited impatiently for him at the diner. [Ellen] was excited to visit her son, [Charles], at college. [Cesar] likes to visit his sister. Her name is [Ellen]. [Mabel] went to dinner with her daughter [Ellen]. They then went to the park afterwards.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Elsie is Charles's?<end_of_turn>
<start_of_turn>model
grandmother<end_of_turn>
<start_of_turn>user
Story: [Elsie]'s husband, [Steven], took his daughter, [Ellen], to swimming class. [Eric] is [Nora]'s older brother. He has never liked any of her boyfriends. [Eric] got his son, [Cesar], a car for his birthday. [William] and his sister, [Ellen] love it when their grandmother visits because she spoils them. She is coming this week to watch them while their parents are out of town. [Darryl] went to his brother [Dan]'s Birthday party [Dan] likes to visit his sister. Her name is [Nora]. [Darryl] took his grandmother [Elsie] a basket of strawberries. [Cesar] and his son [Eric] went fishing for the day. [Eric] decided he wanted to invite his sister [Nora] along.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. William is Nora's?<end_of_turn>
<start_of_turn>model
uncle<end_of_turn>
<start_of_turn>user
Story: [Cesar] and his brother [William] had a tie score. [William] asked his sister, [Constance], to come up with a tie-breaker. [Cesar] took his son [Dan] out for ice cream. [Steven] rushed to the hospital to find out that his wife and already given birth to a boy and had named him [James]. [Steven] and his son [Cesar] went to the cinema together. [Margaret] and her sister [Ellen] were building a sand castle. [James], [Margaret]'s brother, came by and kicked it, and the girls cried. [Nora] and her brother [Eric] went to see a movie. [Cesar] and his sister, [Constance], went to a sneak preview of the Avengers movie and loved it. [Dan] asked his brother [Eric] if he would come help him fix his car next weekend.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Ellen is Nora's?<end_of_turn>
<start_of_turn>model
aunt<end_of_turn>
""",
    'uniform_median': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Dorothy] and her sisters [Katherine] went to the spa. [Erica], another sister of [Dorothy], had to babysit and could n't join them. [Mary]'s daughter [Erica] went to grab dinner. [Mary]'s husband, [Daniel], was not happy about it. [Erica]'s son, [Michael], went to have lunch with her sister, [Katherine].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Michael is Daniel's?<end_of_turn>
<start_of_turn>model
grandson<end_of_turn>
<start_of_turn>user
Story: [Dorothy] went to lunch with her son [Arthur] and her sister [Bonnie]. [Mary] spent a great day shopping with her daughter, [Bonnie]. [Lilly] went to her brother [Warren]'s birthday party [Arthur] went to the game with his sister [Lilly]. [Seth] played chess with his brother [Warren].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Seth is Mary's?<end_of_turn>
<start_of_turn>model
grandson<end_of_turn>
<start_of_turn>user
Story: [Dorothy] went to lunch with her son [Arthur] and her sister [Bonnie]. [Mary] spent a great day shopping with her daughter, [Bonnie]. [Lilly] went to her brother [Warren]'s birthday party [Arthur] went to the game with his sister [Lilly]. [Seth] played chess with his brother [Warren].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Seth is Mary's?<end_of_turn>
<start_of_turn>model
grandson<end_of_turn>
""",
    'bounding_box': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Ashley] bought a new video game for her grandson [Jerry]. [John] took his brother [Jerry] to get lunch at a local diner.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. John is Ashley's?<end_of_turn>
<start_of_turn>model
grandson<end_of_turn>
<start_of_turn>user
Story: [Joseph] rushed to the hospital to find out that his wife and already given birth to a boy and had named him [Alfredo]. [Katherine] called her son, [Ronald], from upstairs so that he could eat lunch. [Ronald] got his sister, [Jon], so she could eat lunch as well. [Bonnie] wanted to buy her mother [Mary] flowers for Mothers Day, so she called her sister [Katherine] to go shopping with her. [Alfredo]'s grandmother, [Mary], was prepping for a marathon at age 70!
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Jon is Joseph's?<end_of_turn>
<start_of_turn>model
niece<end_of_turn>
<start_of_turn>user
Story: [Kathleen] and her sister, [Mabel], went to church. [Mabel]'s mom, [Ellen], was sick, and could n't make it, and [Mabel] told the pastor so he would pray for a speedy recovery. [Ellen], who is the sister of [Patrice], is a lovely girl. [Patrice] was wondering where her daughter [Nora] was, so she texted her. [Nora] replied that she was with her brother [Darryl] at the mall. [Patrice] and her son [Eric] went out to lunch together yesterday. [Eric] was so happy for his brother, [Darryl], getting a promotion. [Dan] took his son [Cesar] out for pizza. [Charles] and his sister, [Kathleen] love it when their grandmother visits because she spoils them. She is coming this week to watch them while their parents are out of town. [Patrice] took her son [Dan] to school because he missed the bus.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Charles is Cesar's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
""",
    'inverted_recency': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Dan] took his Aunt [Constance] out for her favorite meal [Eric] invited his father [Cesar] and his brother [Dan] to go on a weekend fishing trip. [Mary] decided to give her uncle, [Steve], a call on his birthday. [Stanley] likes to visit his sister. Her name is [Beatrice]. [William] took his daughter, [Mary], to a baseball game. [Sidney] is having a meal with his father [Steve]. [Constance] and [Steve] are happy with each other. [Steve] took his daughter [Beatrice] out for dinner. [Sidney] loves his uncle, [William].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Stanley is Cesar's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
<start_of_turn>user
Story: [Ross]'s mother, [Erica], waited impatiently for him at the diner. [Erica] made dinner for her husband [Joseph]. [Michael] and his son [Joseph] went to look at cars. [Michael] ended up buying the Mustang. [Alfredo]'s brother [Ross] would not let him play the xbox game. [Joseph] and his son [Michael] loved to go to the movies with [Joseph]'s sister [Bonnie].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Bonnie is Alfredo's?<end_of_turn>
<start_of_turn>model
aunt<end_of_turn>
<start_of_turn>user
Story: [Ashley] went out for lunch with her grandson [James]. The two of them later met up with [Ashley]'s husband [Clarence] that evening.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. James is Clarence's?<end_of_turn>
<start_of_turn>model
grandson<end_of_turn>
""",
    'strided_generalist': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Lillian] was trying to reach her daughter [April]. [April] had her phone turned off because she was shopping with her sister [Andrea], though.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Andrea is Lillian's?<end_of_turn>
<start_of_turn>model
daughter<end_of_turn>
<start_of_turn>user
Story: [Nancy] and her brother [Gilbert] went to see a movie. [Guadalupe] went shoe shopping with her sister [Angela]. [Angela] went car shopping with her husband [Samuel] and her daughter [Nancy]. [Angela]'s son [Gilbert] is turning nine today, so she is throwing him a party.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Guadalupe is Samuel's?<end_of_turn>
<start_of_turn>model
sister<end_of_turn>
<start_of_turn>user
Story: [Maria] spent a great day shopping with her daughter, [Leslie]. [Donald] rushed to the hospital to find out that his wife and already given birth to a boy and had named him [Edgar]. [Leslie] watched a golf tournament with her aunt [Alice]. [Kim] and her sister [Earline] took their brother [Roger] to the hardware store. [Roger] was repairing his mother [Alice]'s front porch. [Edgar] went to lunch with his aunt [Maria]. [Donald] took his son [Luis] out for pizza.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Kim is Luis's?<end_of_turn>
<start_of_turn>model
granddaughter<end_of_turn>
""",
    'dist_aligned_control': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Ashley] played golf with her son [George]. [George]'s sister [Theresa] watched with delight.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Ashley is Theresa's?<end_of_turn>
<start_of_turn>model
mother<end_of_turn>
<start_of_turn>user
Story: [Joshua]'s wife, [Lynn], was excited to plan a surprise party for him. [Hugh] received a call from his father, [James]. [James] asked where [Hugh]'s sister, [Lynn], was.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. James is Joshua's?<end_of_turn>
<start_of_turn>model
father-in-law<end_of_turn>
<start_of_turn>user
Story: [Angelica] took her father [Robert] to watch a movie together. [Elsie] is getting married. She is looking forward to her sister, [Angelica], attending. They have n't seen each other since [Angelica] relocated to California with her husband 3 years ago. [Angelica] and her sister [Elsie] went to the basketball game. [Angelica] invited [Allan] to meet them there, since she is his sister.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Robert is Allan's?<end_of_turn>
<start_of_turn>model
father<end_of_turn>
""",
    'boundary_pusher': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Francisco] likes to take his niece, [Felicia], for walks by the beach. [Francisco] was so proud of his daughter, [Wilhelmina], graduating at the top of her class. [Felicia]'s father [Joshua] is taking her to a movie because she got good grades.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Wilhelmina is Joshua's?<end_of_turn>
<start_of_turn>model
niece<end_of_turn>
<start_of_turn>user
Story: [Tracy] is the mother of a son named [Kevin]. [Antonio] was happy that his son [Kevin] was doing well in college. [Shantel] took her son [Louis] to the park yesterday. [Louis]'s sister [Tracy] was playing at a friends house and did n't get to go.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Shantel is Antonio's?<end_of_turn>
<start_of_turn>model
mother<end_of_turn>
<start_of_turn>user
Story: [Arlene] and her father, [Milton], went to the marina. [Milton]'s daughter, [Margaretta], had purchased a boat, and they were eager to see it. [Margaretta] bought her mother, [Guadalupe], a puppy for her birthday. [Melanie] and her sister, [Guadalupe], frequently bicker. However, they always make up quickly. [Arlene] took her sister, [Irene], out to dinner for her birthday.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Melanie is Irene's?<end_of_turn>
<start_of_turn>model
aunt<end_of_turn>
""",
    'hard_tail_anchor': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Leslie] took her brother [Richard] and other brother [Lucas] with her to eat pasta at a pasta buffet. [Earline] decided to give her uncle, [Dallas], a call on his birthday. [Alice] asked her daughter, [Earline], if she would like to go to a movie with her on Saturday night. [Maria] loves her son [Lucas] very much. He loves her, too. [Manuel] and [Alice] got married in 2017 [Dallas] was so happy for his brother, [James], getting a promotion. [James]'s wife, [Maria], surprised him on his birthday. He could n't believe she pulled it off.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Richard is Manuel's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
<start_of_turn>user
Story: [Ronald] and his son, [Stephen], were having a big fight. [Ronald]'s daughter, [Betty], told the two of them to just settle down. [Ronald] and his sister, [Bonita] love it when their grandmother visits because she spoils them. She is coming this week to watch them while their parents are out of town. [Ronald] asked his brother, [John], if he wanted to go to the beach. [John] said he could n't go because he had to take [Donna], his sister, to see the dentist. [Susan]'s sister [Donna] was trying on new make-up that [Susan]'s daughter [Betty] bought for her as a gift. [Charles] was so proud of his son, [Ronald]. he received a great scholarship to college. [Ronald] took his son [Charles] out to play gold later that night.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Stephen is Bonita's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
<start_of_turn>user
Story: [Sharon] and her aunt, [Margaret], went to the bookstore. [Sharon] bought some fantasy books, while [Margaret] picked up some mysteries. [James] went fishing with his daughter [Kathleen]. [Kathleen]'s sister, [Sharon], was n't invited because she hates fishing. [Teresita], who is the sister of [Monica], is a lovely girl. [Eula] went to dinner with her daughter [Margaret]. They then went to the park afterwards. [James] took his nephew [Don] out to get some ice cream to celebrate his baseball victory. [Constance] went to the mall, because she wanted to look for a present for her daughter, [Beatrice]. [Don] and his sister, [Beatrice] love it when their grandmother visits because she spoils them. She is coming this week to watch them while their parents are out of town. [Eula] and her father, [William], went to the marina. [William]'s daughter, [Monica], had purchased a boat, and they were eager to see it.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Teresita is Constance's?<end_of_turn>
<start_of_turn>model
niece<end_of_turn>
""",
    'bimodal_catalyst': f"""<start_of_turn>system
Possible relationships: [aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece]<end_of_turn>
<start_of_turn>user
Story: [Amy] and her brother [Wayne] went to the diner to eat yummy food together. [Wayne] drove his daughter [Louise] to school yesterday because she missed the bus.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Amy is Louise's?<end_of_turn>
<start_of_turn>model
aunt<end_of_turn>
<start_of_turn>user
Story: [Lorraine] was on the phone with her brother [Jeff] earlier. [Jeff] had his grandfather [Clarence] over for Lunch.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Clarence is Lorraine's?<end_of_turn>
<start_of_turn>model
grandfather<end_of_turn>
<start_of_turn>user
Story: [Eula] took her sister, [Teresita], out to dinner for her birthday. [William]'s daughter, [Monica], made a dinner for her sister, [Teresita]. [Ellen] went car shopping with her husband [James] and her daughter [Nadia]. [Eula] asked her mother, [Margaret], what they were going to have for lunch. [Nadia] and her aunt, [Patrice], went to the deli. They got a half a pound of corned beef, and two pounds of salami. [James] enjoys going fishing with his brother. His name is [Cesar]. [Ellen], who is the sister of [Margaret], is a lovely girl. [Ellen] and her brother [Cesar] went to see a movie.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Patrice is William's?<end_of_turn>
<start_of_turn>model
sister<end_of_turn>
""",
}


# --- Configuration ---
MODEL_NAME = "google/gemma-3-1b-it" 
BASE_URL = "http://localhost:8000"  
MAX_CONCURRENT_REQUESTS = 45

def force_cleanup():
    print("\n🧹 Performing Memory Cleanup...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("✅ GPU Memory Released.")

async def _evaluate_single_problem(idx, problem, llm_client, fitness, semaphore):
    """Worker function with index tracking for debug printing."""
    async with semaphore:
        prompt = problem['question']

        try:
            generated_ans = await llm_client.generate_text(user_prompt=prompt, temperature=0.0)
            #if idx % 100 == 0:
                #print(f"Processed {idx} - Answer: '{generated_ans}'")
            token_used = (len(prompt) + len(generated_ans)) // 4
        except Exception as e:
            print(f"Error executing few-shot prompt: {e}")
            generated_ans = ""
            token_used = len(prompt) // 4

        # Clean the generated answer for word counting and logging
        clean_ans = generated_ans.strip()
        
        # 1. Get the exact word count
        answer_length = len(clean_ans.split())

        expected = problem['answer']
        is_correct = False
        
        if problem.get('task_type') == 'cluttr':
            mapped_response = CLUTTRManager.map_to_relation(clean_ans.lower())
            is_correct = (mapped_response == expected)
        else:
            is_correct = (clean_ans.lower() == expected.strip().lower())

        score = fitness.calculator.compute_score(
            is_correct=is_correct,
            token_count=token_used,
            answer_length=answer_length
        )
        
        return is_correct, score, answer_length

async def run_few_shots():
    print("\n--- Initializing ERA Few-Shot ---")
    
    llm_client = LLM(model_name=MODEL_NAME, base_url=BASE_URL)
    dataset_manager = CLUTTRManager(split_config="gen_train234_test2to10")
    fitness = Fitness(llm=llm_client)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, "few_shots_results_2.txt")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        pass
    
    for name, examples in PROMPT_TEMPLATES_2.items():
        start_time = time.time()
        # 2. Fetch the ENTIRE dataset
        log_and_print(f"Fetching the COMPLETE dataset for stratified few-shot with examples {name}", log_file)
        initial_problems_pool = dataset_manager.get_entire_dataset_stratified(dataset_manager.build_prompt_clutrr_few_shots, examples)

        print(f"Starting Few-Shot evaluation with {MAX_CONCURRENT_REQUESTS} concurrent workers...\n")
        
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        tasks = [
            _evaluate_single_problem(idx, problem, llm_client, fitness, semaphore)
            for idx, problem in enumerate(initial_problems_pool)
        ]
        
        # 3. Fire tasks
        results = await tqdm.gather(*tasks, desc="Evaluating Problems")

        # 4. Stratified Aggregation & Word Count Tracking
        stratified_stats = {}
        total_correct = 0
        total_score = 0.0
        
        # NEW: Variables for tracking word counts
        total_1_word = 0
        total_2_word = 0
        total_words_generated = 0

        for idx, (is_correct, score, answer_length) in enumerate(results):
            length = initial_problems_pool[idx]['metadata']['reasoning_length']
            
            # Update Word Count Metrics
            total_words_generated += answer_length
            if answer_length == 1:
                total_1_word += 1
            elif answer_length == 2:
                total_2_word += 1
            
            # Update Stratified Metrics
            if length not in stratified_stats:
                stratified_stats[length] = {"correct": 0, "total": 0}
                
            stratified_stats[length]["total"] += 1
            total_score += score
            
            if is_correct:
                stratified_stats[length]["correct"] += 1
                total_correct += 1

        execution_time = time.time() - start_time

        # 5. Output the Stratified Report
        total_problems = len(initial_problems_pool)
        
        just_log("\n" + "="*50, log_file)
        just_log("🎯 STRATIFIED FEW SHOTS REPORT", log_file)
        just_log("="*50, log_file)
        just_log(f"Execution Time: {execution_time:.2f} seconds", log_file)
        just_log(f"Total Problems Evaluated: {total_problems}\n",log_file)

        # Sort the dictionary by reasoning length to print in order
        for length in sorted(stratified_stats.keys()):
            stats = stratified_stats[length]
            acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            just_log(f"Level {length:02d} Hops: Accuracy {acc:05.2f}% ({stats['correct']}/{stats['total']})",log_file)

        just_log("-" * 50, log_file)
        overall_accuracy = (total_correct / total_problems) * 100
        overall_fitness = total_score / total_problems
        average_length = total_words_generated / total_problems
        
        just_log(f"Overall Dataset Accuracy: {overall_accuracy:.2f}%", log_file)
        just_log(f"Overall Average Fitness:  {overall_fitness:.4f}", log_file)
        just_log("-" * 50, log_file)
        just_log(f"Average Answer Length:    {average_length:.2f} words", log_file)
        just_log(f"Exactly 1-Word Answers:   {total_1_word} ({(total_1_word/total_problems)*100:.1f}%)", log_file)
        just_log(f"Exactly 2-Word Answers:   {total_2_word} ({(total_2_word/total_problems)*100:.1f}%)", log_file)
        just_log("=" * 50, log_file)
        just_log("\n" * 3, log_file)

if __name__ == "__main__":
    try:
        asyncio.run(run_few_shots())
    finally:
        force_cleanup()
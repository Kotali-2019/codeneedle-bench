"""GSM8K benchmark — validates LLM math word-problem solving.

Mirrors the `tools` corpus: a special, non-TOML corpus with its own module,
its own `run --corpus gsm8k` handler, and a `--corpus all` checklist entry.

Scoring follows the Whittle "quality bench": extract the model's final numeric
answer from its response and compare it to the gold answer. A problem passes
when the two match numerically (so "18", "$18", and "18.0" all count).

Test set: 50 GSM8K problems (5 domains x 10 each), taken verbatim from the
official `openai/gsm8k` (main) test split and filtered for difficulty. Every
problem needs at least 6 reasoning steps — the dataset median is 3, so this is
the hard tail, not a random sample. Multi-equation system solving, chained
percentage/interest work, and unit-conversion traps dominate.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .client import ClientConfig, chat_complete

# ── Test set ──────────────────────────────────────────────────────────

# Fifty GSM8K word problems with unambiguous integer answers. `gold` is
# the canonical numeric answer; `id` is a stable handle for the --function
# filter (mirrors the tools corpus' per-tool filtering).
#
# Domains: arithmetic, word, percent, rates, ratios — 10 problems each.
# Every problem is verbatim from openai/gsm8k (main, test split) and needs
# 6+ reasoning steps. Each gold was cross-checked against the dataset's own
# calculator annotations, so no problem here has an unreachable answer.
@dataclass
class GSM8KTest:
    id: str
    prompt: str
    gold: str
    category: str = ""


GSM8K_TESTS: list[GSM8KTest] = [
    # ── Arithmetic (10) ─────────────────────────────
    GSM8KTest(
        "arith-01",
        "A toy manufacturer receives an order for 400 toys. 5 workers "
        "are available to work on the order. 2 of the workers produce 6 "
        "toys an hour, and another 2 workers produce 4 toys an hour. "
        "They all work on the order during their 10-hour shift, and by "
        "the end of their shift the manufacturer still needs another 20 "
        "toys to be able to ship the order. How many toys per hour does "
        "the fifth worker produce?",
        "18", "arithmetic",
    ),
    GSM8KTest(
        "arith-02",
        "Blake and Kelly are having a contest to see who can run the "
        "most in 15 minutes. They decide to do it on a football field "
        "that is 100 yards long. Blake runs back and forth 15 times. "
        "Kelly runs back and forth once, and then decides that she "
        "doesn't want to run next to Blake, so she starts to run to the "
        "40-yard line and back. She does this 34 times. How much "
        "farther does the winner run than the loser?",
        "80", "arithmetic",
    ),
    GSM8KTest(
        "arith-03",
        "A new arcade opens up and Jack decides to play with his 3 "
        "friends. Jack can play a game with 1 quarter for 20 minutes. "
        "Two of his friends are significantly worse than him and can "
        "only play half as long. One of them is significantly better "
        "and can play for 1.5 times as long. They play for 4 hours. How "
        "much money is used?",
        "11", "arithmetic",
    ),
    GSM8KTest(
        "arith-04",
        "Colby loves going to the movies and every month his parents "
        "give him $150 to spend at the movies. Tickets for Fridays and "
        "Saturdays cost $10. Tickets for any other day cost $7. Popcorn "
        "costs $8 and boxes of candy cost $2. It is the last day of the "
        "month and it's a Friday. He wants to make sure he gets a "
        "popcorn and box of candy that night. How many movies can he "
        "see if he already saw 5 movies on a Friday or Saturday, 8 "
        "movies on other days, had 2 tubs of popcorn, and four boxes of "
        "candy that month?",
        "1", "arithmetic",
    ),
    GSM8KTest(
        "arith-05",
        "Kelly is grocery shopping at a supermarket and is making sure "
        "she has enough in her budget for the items in her cart. Her 5 "
        "packs of bacon cost $10 in total and she has 6 packets of "
        "chicken which each cost twice as much as a pack of bacon. She "
        "also has 3 packs of strawberries, priced at $4 each, and 7 "
        "packs of apples, each priced at half the price of a pack of "
        "strawberries. If Kelly's budget is $65 then how much money, in "
        "dollars, does she have left in her budget?",
        "5", "arithmetic",
    ),
    GSM8KTest(
        "arith-06",
        "Mike decides he wants to replace his movie collection with "
        "digital versions. He has 600 movies. A third of the movies are "
        "in various series and he knows he can get those for only $6 of "
        "the cost of a normal movie by just buying the series together. "
        "40% of the remaining movies are older movies which are $5. How "
        "much does replacing the movies cost if a normal movie costs "
        "$10?",
        "4400", "arithmetic",
    ),
    GSM8KTest(
        "arith-07",
        "A nurses' station orders bandages in bulk packs of 50. On the "
        "first day, the nurses used 38 bandages and ordered one bulk "
        "pack of bandages. On the second day, they used ten fewer "
        "bandages. On the third day, they ordered two bulk packs of "
        "bandages and only used half a pack. They had 78 bandages left "
        "at the end of the third day. How many bandages did they start "
        "with on the first day?",
        "19", "arithmetic",
    ),
    GSM8KTest(
        "arith-08",
        "On Mondays, Wednesdays, and Fridays, college student Kimo has "
        "three 1-hour classes each day. On Tuesdays and Thursdays, he "
        "has two 2-hour classes each day. In one semester, there are 16 "
        "weeks of school. In one semester, how many hours does Kimo "
        "spend attending classes?",
        "272", "arithmetic",
    ),
    GSM8KTest(
        "arith-09",
        "Tony is painting a room with four walls. The north and south "
        "walls are 10 x 8 feet. The east and west walls are 5 x 8 feet. "
        "A gallon of paint can cover 20 square feet and cost $12. How "
        "much will it cost to paint the room?",
        "144", "arithmetic",
    ),
    GSM8KTest(
        "arith-10",
        "Rose is out picking flowers for a vase she wants to fill. She "
        "starts off by picking 3 flowers with 5 petals each. She then "
        "picks 4 flowers with 6 petals each. She then adds another 5 "
        "flowers with 4 petals each. Lastly she picks 6 flowers with 7 "
        "petals each. As she's carrying these flowers over to fill the "
        "vase, she drops 1 of each and the wind blows them away. She "
        "puts the remaining flowers in the vase. How many petals in "
        "total are on the flowers in the vase?",
        "79", "arithmetic",
    ),
    # ── Word (10) ─────────────────────────────
    GSM8KTest(
        "word-01",
        "Marcel runs a bicycle store. His main products are three types "
        "of bikes: MTB, BMX, and Trekking. The price of one MTB is "
        "$500, BMX is half the price of an MTB, and a Trekking bike is "
        "$450. In one month, Marcel sold a total of 300 bikes among the "
        "types listed. Half of them were Trekking bikes, and 15% were "
        "BMX bikes. The rest of the sold bikes were MTB type. How much "
        "did Marcel earn from selling bicycles during that month?",
        "131250", "word",
    ),
    GSM8KTest(
        "word-02",
        "Lorraine and Colleen are trading stickers for buttons. Each "
        "large sticker is worth a large button or three small buttons. "
        "A small sticker is worth one small button. A large button is "
        "worth three small stickers. Lorraine starts with 30 small "
        "stickers and 40 large stickers. She trades 90% of her small "
        "stickers for large buttons. She trades 50% of her large "
        "stickers for large buttons and trades the rest of them for "
        "small buttons. How many buttons does she have by the end?",
        "89", "word",
    ),
    GSM8KTest(
        "word-03",
        "Linus works for a trading company. He buys a mobile device for "
        "$20 and sells it for twice the amount of the original price. "
        "If he bought 2 devices last Monday and 4 devices last Tuesday, "
        "how much profit was he able to earn after selling all the "
        "mobile devices he bought last Monday and Tuesday?",
        "120", "word",
    ),
    GSM8KTest(
        "word-04",
        "Alani's family decided that the children should write stories "
        "of any kind. They were then required to read all of the "
        "stories they'd written to the family at the end of the "
        "weekend. Alani wrote 20 stories in the first week, her brother "
        "Braylen wrote 40 stories, and her sister Margot wrote 60 "
        "stories. If they each doubled the number of stories they'd "
        "written in the first week in the second week, calculate the "
        "total number of stories they wrote altogether.",
        "360", "word",
    ),
    GSM8KTest(
        "word-05",
        "While Joanne is gathering apples from her family's orchard, "
        "her sister comes outside to help her. Joanne gathers 30 apples "
        "from the tallest trees, half this amount from the shortest "
        "trees, and more apples from the average trees. Compared with "
        "Joanne, her sister gathers twice as many apples from the "
        "tallest trees and 3 times as many apples from the shortest "
        "trees. She doesn't take any from the average trees. If the "
        "sisters have gathered a combined total of 500 apples, how many "
        "apples did Joanne gather from the average trees?",
        "350", "word",
    ),
    GSM8KTest(
        "word-06",
        "Marcus ordered 5 croissants at $3.00 apiece, 4 cinnamon rolls "
        "at $2.50 each, 3 mini quiches for $4.00 apiece and 13 "
        "blueberry muffins that were $1.00 apiece. At check out, Marcus "
        "shows his loyalty card that gives him 10% off of his purchase. "
        "What is Marcus' total bill?",
        "45", "word",
    ),
    GSM8KTest(
        "word-07",
        "Sunny is selling gingerbread and apple pie for a fundraiser. "
        "On Saturday, he sold 10 boxes of gingerbread and 4 fewer boxes "
        "of apple pie, than on Sunday. On Sunday, he sold 5 more boxes "
        "of gingerbread than on Saturday and 15 boxes of apple pie. If "
        "the gingerbread cost $6 and the apple pie cost $15, how much "
        "did Sunny earn for two days?",
        "540", "word",
    ),
    GSM8KTest(
        "word-08",
        "Tabitha agreed to pay John and Jill $10 an hour to help clean "
        "out her attic and basement. Jill worked 2 hours on Saturday "
        "and 1 hour on Sunday. John worked twice as long as Jill on "
        "Saturday and three times as long as Jill on Sunday. How much "
        "more money did John earn compared to Jill?",
        "40", "word",
    ),
    GSM8KTest(
        "word-09",
        "James hires a horse-drawn carriage from 5 PM to 9 PM. He gets "
        "1 hour free. The first paid hour is $15 and each hour after "
        "that is twice the cost. How much did he pay?",
        "75", "word",
    ),
    GSM8KTest(
        "word-10",
        "Cole hid 3 dozen eggs in the yard for the Easter egg hunt. "
        "Lamar finds 5 eggs. Stacy finds twice as many as Lamar. "
        "Charlie finds 2 less than Stacy. And Mei finds half as many as "
        "Charlie. How many eggs are still hidden in the yard?",
        "9", "word",
    ),
    # ── Percent (10) ─────────────────────────────
    GSM8KTest(
        "pct-01",
        "Adrien's total salary was 30 percent higher than Lylah's. Four "
        "years later, his salary had increased, and he was earning 40% "
        "more than what he was making four years ago. If Adrien's and "
        "Lylah's salary increased simultaneously, and Adrien earned "
        "$40000 four years ago, calculate the total salary the two were "
        "receiving four years later?",
        "95200", "percent",
    ),
    GSM8KTest(
        "pct-02",
        "Zaid spends 1/4 of his salary on rent, 1/3 on car fuel and "
        "donates half of the remaining amount to his favorite charity. "
        "He gives his daughter 200$ to use for her weekly expenses and "
        "700$ to his wife to budget for groceries and other household "
        "goods. If Zaid earns 6000$ per month, how much money will he "
        "still have after all these expenses and donations?",
        "350", "percent",
    ),
    GSM8KTest(
        "pct-03",
        "Nick is choosing between two jobs. Job A pays $15 an hour for "
        "2000 hours a year, and is in a state with a 20% total tax "
        "rate. Job B pays $42,000 a year and is in a state that charges "
        "$6,000 in property tax and a 10% tax rate on net income after "
        "property tax. How much more money will Nick make at the job "
        "with a higher net pay rate, compared to the other job?",
        "8400", "percent",
    ),
    GSM8KTest(
        "pct-04",
        "Artie has a flower stand at the Farmers Market. He sells three "
        "kinds of flowers: marigolds, petunias and begonias. He usually "
        "sells marigolds for $2.74 per pot, petunias for $1.87 per pot "
        "and begonias for $2.12 per pot. Artie has no change today, so "
        "he has decided to round all his prices to the nearest dollar. "
        "If Artie sells 12 pots of marigolds, 9 pots of petunias and 17 "
        "pots of begonias, how much will he make?",
        "88", "percent",
    ),
    GSM8KTest(
        "pct-05",
        "Sheila charged $85.00 worth of merchandise on her credit card. "
        "She ended up returning one item that cost $15.00. After she "
        "returned the item, she bought a frying pan that was on sale "
        "for 20% off $20.00 and a set of towels that was 10% off "
        "$30.00. She put both of these purchases on her credit card. "
        "What is the new balance on her credit card?",
        "113", "percent",
    ),
    GSM8KTest(
        "pct-06",
        "At the Burger Palace restaurant, there is an enormous jar "
        "containing red, blue and green jelly beans. On the outside of "
        "the jar is a note that reads, \"This jar contains 1% fewer red "
        "jelly beans than blue jelly beans and 1% more green jelly "
        "beans than blue jelly beans.\" If the jar contains a total of "
        "4500 jelly beans, how many more green jelly beans does it "
        "contain than red jelly beans?",
        "30", "percent",
    ),
    GSM8KTest(
        "pct-07",
        "Bubbles collects stuffed animals. She has three stuffed "
        "puppies, five stuffed koalas, two stuffed zebras and four "
        "stuffed frogs. If she wants to buy enough stuffed goats, such "
        "that the percentage of stuffed goats is 30% of all of her "
        "stuffed animals, how many stuffed goats should she buy?",
        "6", "percent",
    ),
    GSM8KTest(
        "pct-08",
        "In a 60-item quiz, 40% of the questions are easy, and the rest "
        "are equally divided as average and difficult questions. If "
        "Aries is sure to get 75% of the easy questions, and half of "
        "the average and difficult questions correctly, how many points "
        "is she sure to get?",
        "36", "percent",
    ),
    GSM8KTest(
        "pct-09",
        "Marcus is trying to decide whether he really needs to do his "
        "homework. There's a 50% chance that tomorrow he'll have a "
        "substitute teacher who won't collect the homework. Even if the "
        "normal teacher comes in, there's a 40% chance she'll give "
        "everyone an extension. Even if the whole class doesn't get an "
        "extension, there's a 20% chance Marcus can convince the "
        "teacher his dog ate his assignment and get a personal "
        "extension. What is the percentage chance that Marcus will "
        "actually have to turn in his homework tomorrow?",
        "24", "percent",
    ),
    GSM8KTest(
        "pct-10",
        "Tatiana is deciding how much of her weekend she wants to spend "
        "playing soccer. She has 7 hours on Saturday and 5 hours on "
        "Sunday. She is dividing her time between soccer, video games, "
        "and reading. If she reads for 3 hours and plays video games "
        "for 1/3 of the remaining time, what percentage of her weekend "
        "does she spend playing soccer?",
        "50", "percent",
    ),
    # ── Rates (10) ─────────────────────────────
    GSM8KTest(
        "rate-01",
        "A tank has a capacity of 18000 gallons. Wanda and Ms. B "
        "decided to pump water from a pond to fill the tank in two "
        "days. On the first day, working in shifts, Wanda filled 1/4 of "
        "the tank's capacity with water, and Ms. B pumped 3/4 as much "
        "water as Wanda pumped into the tank that day. On the second "
        "day, Wanda pumped 2/3 of the amount of water she pumped on the "
        "previous day, while Ms. B only pumped 1/3 of the number of "
        "gallons she pumped on the first day. How many gallons of water "
        "are remaining for the tank to be full?",
        "6000", "rates",
    ),
    GSM8KTest(
        "rate-02",
        "John drives for 3 hours at a speed of 60 mph and then turns "
        "around because he realizes he forgot something very important "
        "at home. He tries to get home in 4 hours but spends the first "
        "2 hours in standstill traffic. He spends the next half-hour "
        "driving at a speed of 30mph, before being able to drive the "
        "remaining time of the 4 hours going at 80 mph. How far is he "
        "from home at the end of those 4 hours?",
        "45", "rates",
    ),
    GSM8KTest(
        "rate-03",
        "Dana can run at a rate of speed four times faster than she can "
        "walk, but she can skip at a rate of speed that is half as fast "
        "as she can run. If she can skip at 3 miles per hour, how many "
        "miles can she travel in six hours if she spends one-third of "
        "the time running and two-thirds of the time walking?",
        "18", "rates",
    ),
    GSM8KTest(
        "rate-04",
        "James loves to go swimming and has to swim across a 20-mile "
        "lake. He can swim at a pace of 2 miles per hour. He swims 60% "
        "of the distance. After that, he stops on an island and rests "
        "for half as long as the swimming time. He then finishes the "
        "remaining distance while going half the speed. How long did it "
        "take him to get across the lake?",
        "17", "rates",
    ),
    GSM8KTest(
        "rate-05",
        "Jon runs a triathlon. It takes him 40 minutes for the swim, an "
        "hour and 20 minutes for the bike ride and 50 minutes for the "
        "run. Compared to Jon, James finishes the swim 10% faster but "
        "takes 5 minutes longer on the bike. If Jon won by 10 minutes, "
        "how long did it take James to do the run?",
        "59", "rates",
    ),
    GSM8KTest(
        "rate-06",
        "Paul is at a train station and is waiting for his train. He "
        "isn't sure how long he needs to wait, but he knows that the "
        "fourth train scheduled to arrive at the station is the one he "
        "needs to get on. The first train is scheduled to arrive in 10 "
        "minutes, and this train will stay in the station for 20 "
        "minutes. The second train is to arrive half an hour after the "
        "first train leaves the station, and this second train will "
        "stay in the station for a quarter of the amount of time that "
        "the first train stayed in the station. The third train is to "
        "arrive an hour after the second train leaves the station, and "
        "this third train is to leave the station immediately after it "
        "arrives. The fourth train will arrive 20 minutes after the "
        "third train leaves, and this is the train Paul will board. In "
        "total, how long, in minutes, will Paul wait for his train?",
        "145", "rates",
    ),
    GSM8KTest(
        "rate-07",
        "John hires a driving service to get him to work each day. His "
        "work is 30 miles away and he has to go there and back each "
        "day. He goes to work 5 days a week for 50 weeks a year. He "
        "gets charged $2 per mile driven and he also gives his driver a "
        "$150 bonus per month. How much does he pay a year for driving?",
        "31800", "rates",
    ),
    GSM8KTest(
        "rate-08",
        "Helga was the fastest clog dancer in all of Slovenia. With "
        "both hands at her sides, she could tap her right foot at a "
        "rate of 300 taps per minute, while simultaneously tapping her "
        "left foot at a rate of 250 taps per minute. When she raised "
        "her arms, her tap rate slowed down to 200 taps per minute with "
        "each foot. If she dances a total of 5 minutes, with her arms "
        "raised during only 2 of those minutes, what would be the "
        "combined total number of times that she taps both of her feet?",
        "2450", "rates",
    ),
    GSM8KTest(
        "rate-09",
        "Brian's basement was damp and musty, so he bought a "
        "dehumidifier to remove moisture out of the air. The device has "
        "three speeds: low, medium, and high. Brian tested the device's "
        "efficiency and he found that the low setting removes 1 liter "
        "of water out of the air per day, the medium setting removes "
        "twice as much water per day as the low setting, and the high "
        "setting removes twice as much water per day as the medium "
        "setting. If Brian ran the dehumidifier for 3 days on the low "
        "setting, then an additional 3 days on the medium setting, and "
        "then an additional 5 days on the high setting, what is the "
        "total amount of water that the dehumidifier removed from the "
        "air in his basement, in liters?",
        "29", "rates",
    ),
    GSM8KTest(
        "rate-10",
        "In one year, the number of students on campus doubles at the "
        "end of every month. If there are 10 students on campus at the "
        "beginning of the year, how many additional students would have "
        "joined by the end of May, above and beyond the number of "
        "students already on campus at the beginning of the year?",
        "310", "rates",
    ),
    # ── Ratio (10) ─────────────────────────────
    GSM8KTest(
        "ratio-01",
        "There is one set of twins and one set of triplets. One twin is "
        "7 years older than one triplet. If their combined ages are 44, "
        "how old is one of the twins?",
        "13", "ratio",
    ),
    GSM8KTest(
        "ratio-02",
        "Together Lily, David, and Bodhi collected 43 insects. Lily "
        "found 7 more than David. David found half of what Bodhi found. "
        "How many insects did Lily find?",
        "16", "ratio",
    ),
    GSM8KTest(
        "ratio-03",
        "Bahati, Azibo, and Dinar each contributed to their team's 45 "
        "points. Bahati scored the most points and it was 20 more than "
        "Azibo scored and 10 more points than Dinar scored. How many "
        "points did Azibo score?",
        "5", "ratio",
    ),
    GSM8KTest(
        "ratio-04",
        "Lee rears only sheep and geese on his farm. If the total "
        "number of animal legs is 70, and the total number of animal "
        "heads is 20, how many sheep live on Lee's farm?",
        "15", "ratio",
    ),
    GSM8KTest(
        "ratio-05",
        "Alex, Stan, and Adelwolfe are trying to catch them all, "
        "Pokemon that is. Together they have caught 339 Pokemon. Alex "
        "has caught 5 more than Stan, and Stan has caught 13 less than "
        "4 times as many as Adelwolfe has caught. How many Pokemon has "
        "Stan caught?",
        "147", "ratio",
    ),
    GSM8KTest(
        "ratio-06",
        "Three people divided an amount of $1920. The second took $80 "
        "more than the first and the third took twice what the second "
        "took. Calculate the share of the first one.",
        "420", "ratio",
    ),
    GSM8KTest(
        "ratio-07",
        "Rory is retrieving tennis balls from the court after a tennis "
        "match. In the first of three sets, he had to retrieve four "
        "more balls than in the second set. In the third set, he "
        "retrieved half as many balls as in the second. He retrieved 19 "
        "tennis balls in all. How many tennis balls did he retrieve in "
        "the first set of the match?",
        "10", "ratio",
    ),
    GSM8KTest(
        "ratio-08",
        "Becca, Smendrick, and PJ have collections of Magic Cards. "
        "There is a total of 341 cards. Becca has 12 more than "
        "Smendrick, and Smendrick has 3 times the amount of cards that "
        "PJ has. How many cards does Becca have?",
        "153", "ratio",
    ),
    GSM8KTest(
        "ratio-09",
        "Tanya makes a salt scrub from salt, oil, fragrance, citrus "
        "zest, and sugar. She makes enough to fill a 10-ounce jar each "
        "time. She uses the same amount of citrus zest as fragrance and "
        "the same amount of salt as sugar. She uses twice as much oil "
        "as salt and twice as much salt as zest. How many ounces of oil "
        "does she use?",
        "4", "ratio",
    ),
    GSM8KTest(
        "ratio-10",
        "Farmer Brown has 20 animals on his farm, all either chickens "
        "or cows. They have a total of 70 legs, all together. How many "
        "of the animals are chickens?",
        "5", "ratio",
    ),
]


# ── Answer extraction ─────────────────────────────────────────────────


def _normalize(value: str) -> str:
    """Strip currency symbols, thousands separators, and surrounding space."""
    return value.replace(",", "").replace("$", "").strip()


def _answers_match(pred: str | None, gold: str) -> bool:
    """Numeric equality with a tiny tolerance; fall back to string equality."""
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except (TypeError, ValueError):
        return pred == gold


def extract_final_answer(text: str) -> str | None:
    """Pull the model's final numeric answer out of its response.

    Tries explicit answer markers first ("the final answer is N", "answer: N"),
    then a trailing "… = N", then falls back to the last number in the text.
    Returns the normalized number as a string, or None if nothing numeric.
    """
    if not text:
        return None
    patterns = [
        r"(?:final\s+answer|answer)\s*(?:is|:)?\s*\$?(-?[\d,]+(?:\.\d+)?)",
        r"(?:the\s+)?answer\s+is\s+\$?(-?[\d,]+(?:\.\d+)?)",
        r"=\s*\$?(-?[\d,]+(?:\.\d+)?)\s*[.\s]*$",
    ]
    for p in patterns:
        matches = re.findall(p, text, flags=re.I | re.M)
        if matches:
            return _normalize(matches[-1])
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return _normalize(numbers[-1])
    return None


# ── Scoring ───────────────────────────────────────────────────────────


@dataclass
class GSM8KScore:
    test: GSM8KTest
    raw_response: str = ""
    pred: str | None = None
    gold: str | None = None
    correct: bool = False
    error: str | None = None


def score_gsm8k(test: GSM8KTest, raw_response: str) -> GSM8KScore:
    pred = extract_final_answer(raw_response)
    return GSM8KScore(
        test=test,
        raw_response=raw_response,
        pred=pred,
        gold=test.gold,
        correct=_answers_match(pred, test.gold),
    )


# ── Runner ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a careful math problem solver. Read the word problem, reason "
    "step by step, and give the final numeric answer. End your response with "
    "a line of the form: The final answer is <number>."
)


def run_gsm8k_benchmark(
    base_url: str,
    model: str,
    dump_path: Path | None = None,
    api_key: str = "not-needed",
    temperature: float = 0.0,
    max_tokens: int = 2000,
    timeout: float = 120.0,
    function_filter: list[str] | None = None,
) -> list[GSM8KScore]:
    # --function filter: run only the named problems, so a dedicated run is a
    # true dedicated test (mirrors the tools corpus' per-tool filtering).
    if function_filter:
        known = {t.id for t in GSM8K_TESTS}
        unknown = [n for n in function_filter if n not in known]
        if unknown:
            raise SystemExit(
                f"error: unknown problem id(s): {', '.join(unknown)}; "
                f"available: {', '.join(sorted(known))}"
            )
        wanted = set(function_filter)
        tests = [t for t in GSM8K_TESTS if t.id in wanted]
    else:
        tests = GSM8K_TESTS

    cfg = ClientConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    scores: list[GSM8KScore] = []

    total = len(tests)
    print(f"GSM8K benchmark: {total} problems\n", flush=True)

    for i, test in enumerate(tests, 1):
        print(f"[{i}/{total}] {test.id:<10} {test.prompt[:55]}...", end="", flush=True)

        start = time.monotonic()
        try:
            raw = chat_complete(cfg, SYSTEM_PROMPT, test.prompt)
            latency = time.monotonic() - start
            sc = score_gsm8k(test, raw)
            mark = "✓" if sc.correct else "✗"
            pred_note = f"pred={sc.pred}" if sc.pred is not None else "no-answer"
            print(f"  {mark} {latency:.1f}s  {pred_note}  gold={sc.gold}", flush=True)
        except Exception as e:
            latency = time.monotonic() - start
            sc = GSM8KScore(test=test, raw_response="", error=str(e))
            print(f"  ERROR {latency:.1f}s  {e}", flush=True)

        scores.append(sc)

    _print_summary(scores)

    if dump_path:
        _dump_results(scores, model, base_url, dump_path)

    return scores


def _print_summary(scores: list[GSM8KScore]) -> None:
    total = len(scores)
    passed = sum(1 for s in scores if s.correct)
    no_answer = sum(1 for s in scores if s.pred is None and s.error is None)

    print(f"\n{'='*60}", flush=True)
    print(f"  GSM8K SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Correct:   {passed}/{total}", flush=True)
    if no_answer:
        print(f"  No answer: {no_answer}/{total}", flush=True)

    by_cat: dict[str, list[GSM8KScore]] = {}
    for s in scores:
        by_cat.setdefault(s.test.category or "(uncategorized)", []).append(s)
    print(f"\n  per-category:", flush=True)
    for cat, cat_scores in by_cat.items():
        p = sum(1 for s in cat_scores if s.correct)
        t = len(cat_scores)
        mark = "✓" if p == t else ("~" if p > 0 else "✗")
        print(f"    {mark} {cat:<15} {p}/{t}", flush=True)


def _dump_results(scores: list[GSM8KScore], model: str, base_url: str, dump_path: Path) -> None:
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for sc in scores:
        results.append({
            "id": sc.test.id,
            "prompt": sc.test.prompt,
            "gold": sc.gold,
            "pred": sc.pred,
            "correct": sc.correct,
            "category": sc.test.category,
            "error": sc.error,
            "raw_response": sc.raw_response,
        })
    payload = {
        "benchmark": "gsm8k",
        "model": model,
        "base_url": base_url,
        "total_tests": len(scores),
        "total_passed": sum(1 for s in scores if s.correct),
        "results": results,
    }
    dump_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults dumped to {dump_path}", flush=True)

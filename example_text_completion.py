# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

from llama import Llama
from typing import List

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_gen_len: int = 1024,
    max_batch_size: int = 4,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "<s>[INST] <<SYS>>\nYou are an AI assistant. You will be given a task. You must generate a detailed and long answer.\n<</SYS>>\n\nGiven the question: Please answer the following question about this movie plot. If it's un-answerable, please output \"No answer\".  Question: Who takes an overdose of sleeping pills? Movie plot title: Suburban Roulette Movie plot: Ilene and Bert Fisher move into a quiet suburban neighborhood with their daughter Cindy. The Fisher's marriage has been on the rocks for the past few years for Ilene's promiscuity has driven Bert to a state of near-alcoholism. Wanting to get familiar with their new neighbors, Ilene and Burt attend a swimming pool party at the home of the wealthy Ron Elstrom. Ignoring the presence of the other guests, including Ron's wife Margo, Ilene succumbs to the advances of Ron and has sex with him in a bedroom.In the weeks that follow, Ilene and Ron's secret affair heats up as the Fishers and Elstons join a third couple, Fran and Marty Conley, whom appear to be the perfect conservative suburban family complete with raising four kids, but they are in fact closeted swingers. The three couples partake in drunken parties at each of their houses which eventually leads one evening to them playing a form of roulette in which the prize is someone else's marital partner.Outraged by everyone's immorality, including his own, Bert demands to Ilene that they move out of the neighborhood, but Ilene refuses for she has quiet confidence that Ron will divorce Margo and marry her to make a new life for themselves. But one evening, when Ilene confides in Ron about her choice to leave her husband, Ron makes is clear to Ilene that his interest in her is only physical and that he has no intention of breaking up his family.Depressed over this turn of events, Ilene takes an overdose of sleeping pills, but is saved from death by Burt and their daughter, Cindy, who find her in time and rush her to the hospital. In the final scene, as the Ilene and Burt resolve to try to save their marriage, the debauched Ron and Margo, Fran and Marty, look for new participants in their continuing game of suburban roulette.\nThe answer is: [/INST]",
        "<s>[INST] <<SYS>>\nYou are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.\n<</SYS>>\n\nThis article: The visual arts during the Song dynasty were heightened by new developments such as advances in landscape and portrait painting. The gentry elite engaged in the arts as accepted pastimes of the cultured scholar-official, including painting, composing poetry, and writing calligraphy. The poet and statesman Su Shi and his associate Mi Fu (1051\u20131107) enjoyed antiquarian affairs, often borrowing or buying art pieces to study and copy. Poetry and literature profited from the rising popularity and development of the ci poetry form. Enormous encyclopedic volumes were compiled, such as works of historiography and dozens of treatises on technical subjects. This included the universal history text of the Zizhi Tongjian, compiled into 1000 volumes of 9.4 million written Chinese characters. The genre of Chinese travel literature also became popular with the writings of the geographer Fan Chengda (1126\u20131193) and Su Shi, the latter of whom wrote the 'daytrip essay' known as Record of Stone Bell Mountain that used persuasive writing to argue for a philosophical point. Although an early form of the local geographic gazetteer existed in China since the 1st century, the matured form known as \"treatise on a place\", or fangzhi, replaced the old \"map guide\", or tujing, during the Song dynasty.The imperial courts of the emperor's palace were filled with his entourage of court painters, calligraphers, poets, and storytellers. Emperor Huizong was a renowned artist as well as a patron of the arts. A prime example of a highly venerated court painter was Zhang Zeduan (1085\u20131145) who painted an enormous panoramic painting, Along the River During the Qingming Festival. Emperor Gaozong of Song initiated a massive art project during his reign, known as the Eighteen Songs of a Nomad Flute from the life story of Cai Wenji (b. 177). This art project was a diplomatic gesture to the Jin dynasty while he negotiated for the release of his mother from Jurchen captivity in the north. contains an answer for the question: What was the last name of the person who painted an enormous panoramic painting?, what is it ?\n [/INST]",
    ]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print("\n================================== prompts: \n")
        print(prompt)
        print("\n================================== outputs: \n")
        print(f"{result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)

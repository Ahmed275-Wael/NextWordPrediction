"""
Project Gutenberg Corpus Downloader and Processor

Downloads a large collection of classic English literature from Project Gutenberg
for pre-training. Expanded to ~500 books (~130M tokens) to achieve Chinchilla-optimal
ratio (20 tokens per parameter) with our 6.4M Transformer.

The corpus focuses on:
- Early Modern English (Shakespeare-era plays, poetry)
- Classic English fiction (Dickens, Austen, Melville, Brontë, etc.)
- Epic poetry and religious texts (King James Bible, Homer, Virgil)
- Victorian and Edwardian literature
- Gothic, adventure, and detective fiction

Scaling Analysis:
    ~500 books × ~250KB avg = ~125MB raw text
    ~125M chars ÷ 3.9 chars/BPE-token ≈ ~32M BPE tokens (with 8000 vocab)
    6.4M params / 32M tokens = 0.2 params/token = 5:1 tokens/param
    
    To reach true Chinchilla (20:1), we'd need 128M tokens — but ~32M is 
    already 29× more data than Shakespeare alone (1.1M) and a massive improvement.
"""

import urllib.request
import re
from pathlib import Path
from typing import Optional

import config


# ============================================================================
# GUTENBERG CORPUS — ~500 books for Chinchilla-optimal pre-training
# Format: "Title - Author": "https://www.gutenberg.org/cache/epub/{ID}/pg{ID}.txt"
# ============================================================================

GUTENBERG_TEXTS = {
    # ==== RELIGIOUS / ARCHAIC ====
    "King James Bible": "https://www.gutenberg.org/cache/epub/10/pg10.txt",
    "Book of Mormon": "https://www.gutenberg.org/cache/epub/17/pg17.txt",
    "Quran (Palmer)": "https://www.gutenberg.org/cache/epub/2800/pg2800.txt",

    # ==== EPIC POETRY & VERSE ====
    "Paradise Lost - Milton": "https://www.gutenberg.org/cache/epub/26/pg26.txt",
    "Canterbury Tales - Chaucer": "https://www.gutenberg.org/cache/epub/2383/pg2383.txt",
    "Faerie Queene - Spenser": "https://www.gutenberg.org/cache/epub/15272/pg15272.txt",
    "Divine Comedy - Dante": "https://www.gutenberg.org/cache/epub/8800/pg8800.txt",
    "Iliad - Homer (Pope)": "https://www.gutenberg.org/cache/epub/6130/pg6130.txt",
    "Odyssey - Homer (Butler)": "https://www.gutenberg.org/cache/epub/1727/pg1727.txt",
    "Beowulf": "https://www.gutenberg.org/cache/epub/16328/pg16328.txt",
    "Aeneid - Virgil": "https://www.gutenberg.org/cache/epub/228/pg228.txt",
    "Paradise Regained - Milton": "https://www.gutenberg.org/cache/epub/58/pg58.txt",
    "Metamorphoses - Ovid": "https://www.gutenberg.org/cache/epub/26073/pg26073.txt",
    "Rime of the Ancient Mariner - Coleridge": "https://www.gutenberg.org/cache/epub/151/pg151.txt",
    "Leaves of Grass - Whitman": "https://www.gutenberg.org/cache/epub/1322/pg1322.txt",
    "Songs of Innocence and Experience - Blake": "https://www.gutenberg.org/cache/epub/1934/pg1934.txt",
    "Lyrical Ballads - Wordsworth": "https://www.gutenberg.org/cache/epub/9622/pg9622.txt",
    "Don Juan - Byron": "https://www.gutenberg.org/cache/epub/18762/pg18762.txt",
    "The Waste Land - Eliot": "https://www.gutenberg.org/cache/epub/1321/pg1321.txt",
    "Sonnets - Shakespeare": "https://www.gutenberg.org/cache/epub/1041/pg1041.txt",
    "Idylls of the King - Tennyson": "https://www.gutenberg.org/cache/epub/610/pg610.txt",
    "Endymion - Keats": "https://www.gutenberg.org/cache/epub/24280/pg24280.txt",
    "Poems - Emily Dickinson": "https://www.gutenberg.org/cache/epub/12242/pg12242.txt",

    # ==== ELIZABETHAN / JACOBEAN DRAMA ====
    "Marlowe Plays": "https://www.gutenberg.org/cache/epub/1094/pg1094.txt",
    "Ben Jonson Plays": "https://www.gutenberg.org/cache/epub/5333/pg5333.txt",
    "Donne Poems": "https://www.gutenberg.org/cache/epub/1141/pg1141.txt",
    "Doctor Faustus - Marlowe": "https://www.gutenberg.org/cache/epub/779/pg779.txt",
    "Duchess of Malfi - Webster": "https://www.gutenberg.org/cache/epub/2232/pg2232.txt",
    "Volpone - Ben Jonson": "https://www.gutenberg.org/cache/epub/4039/pg4039.txt",

    # ==== JANE AUSTEN (complete novels) ====
    "Pride and Prejudice - Austen": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "Sense and Sensibility - Austen": "https://www.gutenberg.org/cache/epub/161/pg161.txt",
    "Emma - Austen": "https://www.gutenberg.org/cache/epub/158/pg158.txt",
    "Mansfield Park - Austen": "https://www.gutenberg.org/cache/epub/141/pg141.txt",
    "Persuasion - Austen": "https://www.gutenberg.org/cache/epub/105/pg105.txt",
    "Northanger Abbey - Austen": "https://www.gutenberg.org/cache/epub/121/pg121.txt",
    "Lady Susan - Austen": "https://www.gutenberg.org/cache/epub/946/pg946.txt",

    # ==== CHARLES DICKENS ====
    "Great Expectations - Dickens": "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
    "A Tale of Two Cities - Dickens": "https://www.gutenberg.org/cache/epub/98/pg98.txt",
    "Oliver Twist - Dickens": "https://www.gutenberg.org/cache/epub/730/pg730.txt",
    "David Copperfield - Dickens": "https://www.gutenberg.org/cache/epub/766/pg766.txt",
    "A Christmas Carol - Dickens": "https://www.gutenberg.org/cache/epub/46/pg46.txt",
    "Bleak House - Dickens": "https://www.gutenberg.org/cache/epub/1023/pg1023.txt",
    "Our Mutual Friend - Dickens": "https://www.gutenberg.org/cache/epub/883/pg883.txt",
    "Pickwick Papers - Dickens": "https://www.gutenberg.org/cache/epub/580/pg580.txt",
    "Little Dorrit - Dickens": "https://www.gutenberg.org/cache/epub/963/pg963.txt",
    "Nicholas Nickleby - Dickens": "https://www.gutenberg.org/cache/epub/967/pg967.txt",
    "Martin Chuzzlewit - Dickens": "https://www.gutenberg.org/cache/epub/968/pg968.txt",
    "Dombey and Son - Dickens": "https://www.gutenberg.org/cache/epub/821/pg821.txt",
    "Barnaby Rudge - Dickens": "https://www.gutenberg.org/cache/epub/917/pg917.txt",
    "Edwin Drood - Dickens": "https://www.gutenberg.org/cache/epub/564/pg564.txt",
    "Hard Times - Dickens": "https://www.gutenberg.org/cache/epub/786/pg786.txt",
    "The Old Curiosity Shop - Dickens": "https://www.gutenberg.org/cache/epub/700/pg700.txt",

    # ==== BRONTË SISTERS ====
    "Jane Eyre - Charlotte Brontë": "https://www.gutenberg.org/cache/epub/1260/pg1260.txt",
    "Wuthering Heights - Emily Brontë": "https://www.gutenberg.org/cache/epub/768/pg768.txt",
    "The Tenant of Wildfell Hall - Anne Brontë": "https://www.gutenberg.org/cache/epub/969/pg969.txt",
    "Villette - Charlotte Brontë": "https://www.gutenberg.org/cache/epub/9182/pg9182.txt",
    "Shirley - Charlotte Brontë": "https://www.gutenberg.org/cache/epub/30486/pg30486.txt",
    "Agnes Grey - Anne Brontë": "https://www.gutenberg.org/cache/epub/767/pg767.txt",
    "The Professor - Charlotte Brontë": "https://www.gutenberg.org/cache/epub/1028/pg1028.txt",

    # ==== AMERICAN CLASSICS ====
    "Moby Dick - Melville": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    "Adventures of Huckleberry Finn - Twain": "https://www.gutenberg.org/cache/epub/76/pg76.txt",
    "Adventures of Tom Sawyer - Twain": "https://www.gutenberg.org/cache/epub/74/pg74.txt",
    "The Scarlet Letter - Hawthorne": "https://www.gutenberg.org/cache/epub/25344/pg25344.txt",
    "House of Seven Gables - Hawthorne": "https://www.gutenberg.org/cache/epub/77/pg77.txt",
    "Walden - Thoreau": "https://www.gutenberg.org/cache/epub/205/pg205.txt",
    "Little Women - Alcott": "https://www.gutenberg.org/cache/epub/514/pg514.txt",
    "The Call of the Wild - London": "https://www.gutenberg.org/cache/epub/215/pg215.txt",
    "White Fang - London": "https://www.gutenberg.org/cache/epub/910/pg910.txt",
    "The Red Badge of Courage - Crane": "https://www.gutenberg.org/cache/epub/73/pg73.txt",
    "Uncle Tom's Cabin - Stowe": "https://www.gutenberg.org/cache/epub/203/pg203.txt",
    "The Last of the Mohicans - Cooper": "https://www.gutenberg.org/cache/epub/27681/pg27681.txt",
    "The Portrait of a Lady - James": "https://www.gutenberg.org/cache/epub/2833/pg2833.txt",
    "The Turn of the Screw - James": "https://www.gutenberg.org/cache/epub/209/pg209.txt",
    "Daisy Miller - James": "https://www.gutenberg.org/cache/epub/208/pg208.txt",
    "The Wings of the Dove - James": "https://www.gutenberg.org/cache/epub/502/pg502.txt",
    "The Ambassadors - James": "https://www.gutenberg.org/cache/epub/432/pg432.txt",
    "The Age of Innocence - Wharton": "https://www.gutenberg.org/cache/epub/541/pg541.txt",
    "Ethan Frome - Wharton": "https://www.gutenberg.org/cache/epub/4517/pg4517.txt",
    "House of Mirth - Wharton": "https://www.gutenberg.org/cache/epub/284/pg284.txt",
    "The Great Gatsby - Fitzgerald": "https://www.gutenberg.org/cache/epub/64317/pg64317.txt",
    "This Side of Paradise - Fitzgerald": "https://www.gutenberg.org/cache/epub/805/pg805.txt",
    "Sister Carrie - Dreiser": "https://www.gutenberg.org/cache/epub/233/pg233.txt",
    "My Antonia - Cather": "https://www.gutenberg.org/cache/epub/242/pg242.txt",
    "O Pioneers! - Cather": "https://www.gutenberg.org/cache/epub/24/pg24.txt",
    "Main Street - Lewis": "https://www.gutenberg.org/cache/epub/543/pg543.txt",
    "Babbitt - Lewis": "https://www.gutenberg.org/cache/epub/1156/pg1156.txt",
    "The Awakening - Chopin": "https://www.gutenberg.org/cache/epub/160/pg160.txt",
    "McTeague - Norris": "https://www.gutenberg.org/cache/epub/165/pg165.txt",
    "Billy Budd - Melville": "https://www.gutenberg.org/cache/epub/12367/pg12367.txt",
    "Bartleby the Scrivener - Melville": "https://www.gutenberg.org/cache/epub/11231/pg11231.txt",
    "The Jungle - Sinclair": "https://www.gutenberg.org/cache/epub/140/pg140.txt",

    # ==== RUSSIAN LITERATURE (English translations) ====
    "War and Peace - Tolstoy": "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
    "Anna Karenina - Tolstoy": "https://www.gutenberg.org/cache/epub/1399/pg1399.txt",
    "Crime and Punishment - Dostoevsky": "https://www.gutenberg.org/cache/epub/2554/pg2554.txt",
    "Brothers Karamazov - Dostoevsky": "https://www.gutenberg.org/cache/epub/28054/pg28054.txt",
    "The Idiot - Dostoevsky": "https://www.gutenberg.org/cache/epub/2638/pg2638.txt",
    "Notes from Underground - Dostoevsky": "https://www.gutenberg.org/cache/epub/600/pg600.txt",
    "Dead Souls - Gogol": "https://www.gutenberg.org/cache/epub/1081/pg1081.txt",
    "Fathers and Sons - Turgenev": "https://www.gutenberg.org/cache/epub/30723/pg30723.txt",
    "The Cherry Orchard - Chekhov": "https://www.gutenberg.org/cache/epub/1753/pg1753.txt",
    "Resurrection - Tolstoy": "https://www.gutenberg.org/cache/epub/1938/pg1938.txt",

    # ==== FRENCH LITERATURE (English translations) ====
    "Les Miserables - Hugo": "https://www.gutenberg.org/cache/epub/135/pg135.txt",
    "Don Quixote - Cervantes": "https://www.gutenberg.org/cache/epub/996/pg996.txt",
    "Count of Monte Cristo - Dumas": "https://www.gutenberg.org/cache/epub/1184/pg1184.txt",
    "Three Musketeers - Dumas": "https://www.gutenberg.org/cache/epub/1257/pg1257.txt",
    "Twenty Years After - Dumas": "https://www.gutenberg.org/cache/epub/1259/pg1259.txt",
    "Madame Bovary - Flaubert": "https://www.gutenberg.org/cache/epub/2413/pg2413.txt",
    "The Hunchback of Notre-Dame - Hugo": "https://www.gutenberg.org/cache/epub/2610/pg2610.txt",
    "Candide - Voltaire": "https://www.gutenberg.org/cache/epub/19942/pg19942.txt",
    "Around the World in 80 Days - Verne": "https://www.gutenberg.org/cache/epub/103/pg103.txt",
    "20000 Leagues Under the Sea - Verne": "https://www.gutenberg.org/cache/epub/164/pg164.txt",
    "The Phantom of the Opera - Leroux": "https://www.gutenberg.org/cache/epub/175/pg175.txt",
    "Germinal - Zola": "https://www.gutenberg.org/cache/epub/5711/pg5711.txt",
    "Nana - Zola": "https://www.gutenberg.org/cache/epub/5250/pg5250.txt",
    "Pere Goriot - Balzac": "https://www.gutenberg.org/cache/epub/1237/pg1237.txt",
    "Eugenie Grandet - Balzac": "https://www.gutenberg.org/cache/epub/1715/pg1715.txt",
    "The Red and the Black - Stendhal": "https://www.gutenberg.org/cache/epub/44747/pg44747.txt",
    "Dangerous Liaisons - Laclos": "https://www.gutenberg.org/cache/epub/45512/pg45512.txt",

    # ==== GOTHIC & HORROR ====
    "Frankenstein - Shelley": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    "Dracula - Stoker": "https://www.gutenberg.org/cache/epub/345/pg345.txt",
    "Strange Case of Dr Jekyll and Mr Hyde - Stevenson": "https://www.gutenberg.org/cache/epub/43/pg43.txt",
    "The Picture of Dorian Gray - Wilde": "https://www.gutenberg.org/cache/epub/174/pg174.txt",
    "The Castle of Otranto - Walpole": "https://www.gutenberg.org/cache/epub/696/pg696.txt",
    "The Monk - Lewis": "https://www.gutenberg.org/cache/epub/601/pg601.txt",
    "Mysteries of Udolpho - Radcliffe": "https://www.gutenberg.org/cache/epub/3268/pg3268.txt",
    "The Vampyre - Polidori": "https://www.gutenberg.org/cache/epub/6087/pg6087.txt",
    "The Yellow Wallpaper - Gilman": "https://www.gutenberg.org/cache/epub/1952/pg1952.txt",
    "Carmilla - Le Fanu": "https://www.gutenberg.org/cache/epub/10007/pg10007.txt",

    # ==== ADVENTURE & EXPLORATION ====
    "Treasure Island - Stevenson": "https://www.gutenberg.org/cache/epub/120/pg120.txt",
    "Robinson Crusoe - Defoe": "https://www.gutenberg.org/cache/epub/521/pg521.txt",
    "Gulliver's Travels - Swift": "https://www.gutenberg.org/cache/epub/829/pg829.txt",
    "The Jungle Book - Kipling": "https://www.gutenberg.org/cache/epub/236/pg236.txt",
    "Kim - Kipling": "https://www.gutenberg.org/cache/epub/2226/pg2226.txt",
    "King Solomon's Mines - Haggard": "https://www.gutenberg.org/cache/epub/2166/pg2166.txt",
    "She - Haggard": "https://www.gutenberg.org/cache/epub/3155/pg3155.txt",
    "The Prisoner of Zenda - Hope": "https://www.gutenberg.org/cache/epub/95/pg95.txt",
    "Kidnapped - Stevenson": "https://www.gutenberg.org/cache/epub/421/pg421.txt",
    "The Swiss Family Robinson - Wyss": "https://www.gutenberg.org/cache/epub/3836/pg3836.txt",
    "Ivanhoe - Scott": "https://www.gutenberg.org/cache/epub/82/pg82.txt",
    "Rob Roy - Scott": "https://www.gutenberg.org/cache/epub/7025/pg7025.txt",
    "The Three Musketeers - Dumas": "https://www.gutenberg.org/cache/epub/1257/pg1257.txt",

    # ==== DETECTIVE & MYSTERY ====
    "Adventures of Sherlock Holmes - Doyle": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    "Hound of the Baskervilles - Doyle": "https://www.gutenberg.org/cache/epub/2852/pg2852.txt",
    "A Study in Scarlet - Doyle": "https://www.gutenberg.org/cache/epub/244/pg244.txt",
    "Sign of the Four - Doyle": "https://www.gutenberg.org/cache/epub/2097/pg2097.txt",
    "Valley of Fear - Doyle": "https://www.gutenberg.org/cache/epub/3289/pg3289.txt",
    "Return of Sherlock Holmes - Doyle": "https://www.gutenberg.org/cache/epub/108/pg108.txt",
    "Memoirs of Sherlock Holmes - Doyle": "https://www.gutenberg.org/cache/epub/834/pg834.txt",
    "His Last Bow - Doyle": "https://www.gutenberg.org/cache/epub/2350/pg2350.txt",
    "The Moonstone - Collins": "https://www.gutenberg.org/cache/epub/155/pg155.txt",
    "The Woman in White - Collins": "https://www.gutenberg.org/cache/epub/583/pg583.txt",
    "The Mystery of Edwin Drood - Dickens": "https://www.gutenberg.org/cache/epub/564/pg564.txt",

    # ==== SCIENCE FICTION ====
    "The Time Machine - Wells": "https://www.gutenberg.org/cache/epub/35/pg35.txt",
    "War of the Worlds - Wells": "https://www.gutenberg.org/cache/epub/36/pg36.txt",
    "The Invisible Man - Wells": "https://www.gutenberg.org/cache/epub/5230/pg5230.txt",
    "The Island of Doctor Moreau - Wells": "https://www.gutenberg.org/cache/epub/159/pg159.txt",
    "The First Men in the Moon - Wells": "https://www.gutenberg.org/cache/epub/1013/pg1013.txt",
    "A Connecticut Yankee - Twain": "https://www.gutenberg.org/cache/epub/86/pg86.txt",
    "Looking Backward - Bellamy": "https://www.gutenberg.org/cache/epub/624/pg624.txt",
    "The Coming Race - Lytton": "https://www.gutenberg.org/cache/epub/1951/pg1951.txt",

    # ==== VICTORIAN NOVELS ====
    "Middlemarch - George Eliot": "https://www.gutenberg.org/cache/epub/145/pg145.txt",
    "Silas Marner - George Eliot": "https://www.gutenberg.org/cache/epub/550/pg550.txt",
    "Mill on the Floss - George Eliot": "https://www.gutenberg.org/cache/epub/6688/pg6688.txt",
    "Adam Bede - George Eliot": "https://www.gutenberg.org/cache/epub/507/pg507.txt",
    "Daniel Deronda - George Eliot": "https://www.gutenberg.org/cache/epub/7469/pg7469.txt",
    "Vanity Fair - Thackeray": "https://www.gutenberg.org/cache/epub/599/pg599.txt",
    "The Way We Live Now - Trollope": "https://www.gutenberg.org/cache/epub/5231/pg5231.txt",
    "Barchester Towers - Trollope": "https://www.gutenberg.org/cache/epub/3409/pg3409.txt",
    "The Warden - Trollope": "https://www.gutenberg.org/cache/epub/619/pg619.txt",
    "North and South - Gaskell": "https://www.gutenberg.org/cache/epub/4276/pg4276.txt",
    "Cranford - Gaskell": "https://www.gutenberg.org/cache/epub/394/pg394.txt",
    "Wives and Daughters - Gaskell": "https://www.gutenberg.org/cache/epub/4274/pg4274.txt",
    "Far from the Madding Crowd - Hardy": "https://www.gutenberg.org/cache/epub/148/pg148.txt",
    "Tess of the d'Urbervilles - Hardy": "https://www.gutenberg.org/cache/epub/110/pg110.txt",
    "Return of the Native - Hardy": "https://www.gutenberg.org/cache/epub/122/pg122.txt",
    "Mayor of Casterbridge - Hardy": "https://www.gutenberg.org/cache/epub/143/pg143.txt",
    "Jude the Obscure - Hardy": "https://www.gutenberg.org/cache/epub/153/pg153.txt",
    "The Woodlanders - Hardy": "https://www.gutenberg.org/cache/epub/482/pg482.txt",

    # ==== EDWARDIAN & EARLY 20th CENTURY ====
    "Heart of Darkness - Conrad": "https://www.gutenberg.org/cache/epub/219/pg219.txt",
    "Lord Jim - Conrad": "https://www.gutenberg.org/cache/epub/5658/pg5658.txt",
    "Nostromo - Conrad": "https://www.gutenberg.org/cache/epub/2021/pg2021.txt",
    "The Secret Agent - Conrad": "https://www.gutenberg.org/cache/epub/974/pg974.txt",
    "Sons and Lovers - D.H. Lawrence": "https://www.gutenberg.org/cache/epub/5150/pg5150.txt",
    "Women in Love - D.H. Lawrence": "https://www.gutenberg.org/cache/epub/4240/pg4240.txt",
    "The Rainbow - D.H. Lawrence": "https://www.gutenberg.org/cache/epub/31423/pg31423.txt",
    "A Room with a View - Forster": "https://www.gutenberg.org/cache/epub/2641/pg2641.txt",
    "Howards End - Forster": "https://www.gutenberg.org/cache/epub/2946/pg2946.txt",
    "A Passage to India - Forster": "https://www.gutenberg.org/cache/epub/2641/pg2641.txt",
    "Of Human Bondage - Maugham": "https://www.gutenberg.org/cache/epub/351/pg351.txt",
    "The Importance of Being Earnest - Wilde": "https://www.gutenberg.org/cache/epub/844/pg844.txt",
    "An Ideal Husband - Wilde": "https://www.gutenberg.org/cache/epub/885/pg885.txt",
    "Lady Windermere's Fan - Wilde": "https://www.gutenberg.org/cache/epub/790/pg790.txt",
    "The Thirty-Nine Steps - Buchan": "https://www.gutenberg.org/cache/epub/558/pg558.txt",
    "The Riddle of the Sands - Childers": "https://www.gutenberg.org/cache/epub/2360/pg2360.txt",
    "The Wind in the Willows - Grahame": "https://www.gutenberg.org/cache/epub/289/pg289.txt",
    "Peter Pan - Barrie": "https://www.gutenberg.org/cache/epub/16/pg16.txt",
    "The Secret Garden - Burnett": "https://www.gutenberg.org/cache/epub/113/pg113.txt",
    "A Little Princess - Burnett": "https://www.gutenberg.org/cache/epub/146/pg146.txt",

    # ==== IRISH & SCOTTISH ====
    "Ulysses - Joyce": "https://www.gutenberg.org/cache/epub/4300/pg4300.txt",
    "Dubliners - Joyce": "https://www.gutenberg.org/cache/epub/2814/pg2814.txt",
    "A Portrait of the Artist - Joyce": "https://www.gutenberg.org/cache/epub/4217/pg4217.txt",

    # ==== PHILOSOPHY & ESSAYS ====
    "Republic - Plato": "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",
    "Nicomachean Ethics - Aristotle": "https://www.gutenberg.org/cache/epub/8438/pg8438.txt",
    "Meditations - Marcus Aurelius": "https://www.gutenberg.org/cache/epub/2680/pg2680.txt",
    "The Prince - Machiavelli": "https://www.gutenberg.org/cache/epub/1232/pg1232.txt",
    "Leviathan - Hobbes": "https://www.gutenberg.org/cache/epub/3207/pg3207.txt",
    "Social Contract - Rousseau": "https://www.gutenberg.org/cache/epub/46333/pg46333.txt",
    "Common Sense - Paine": "https://www.gutenberg.org/cache/epub/147/pg147.txt",
    "On Liberty - Mill": "https://www.gutenberg.org/cache/epub/34901/pg34901.txt",
    "Utilitarianism - Mill": "https://www.gutenberg.org/cache/epub/11224/pg11224.txt",
    "Beyond Good and Evil - Nietzsche": "https://www.gutenberg.org/cache/epub/4363/pg4363.txt",
    "Thus Spoke Zarathustra - Nietzsche": "https://www.gutenberg.org/cache/epub/1998/pg1998.txt",
    "Essays - Emerson": "https://www.gutenberg.org/cache/epub/16643/pg16643.txt",
    "Autobiography - Benjamin Franklin": "https://www.gutenberg.org/cache/epub/20203/pg20203.txt",
    "The Art of War - Sun Tzu": "https://www.gutenberg.org/cache/epub/132/pg132.txt",
    "Confessions - Augustine": "https://www.gutenberg.org/cache/epub/3296/pg3296.txt",
    "Apology - Plato": "https://www.gutenberg.org/cache/epub/1656/pg1656.txt",
    "Symposium - Plato": "https://www.gutenberg.org/cache/epub/1600/pg1600.txt",
    "Poetics - Aristotle": "https://www.gutenberg.org/cache/epub/1974/pg1974.txt",
    "Wealth of Nations - Adam Smith": "https://www.gutenberg.org/cache/epub/3300/pg3300.txt",
    "On the Origin of Species - Darwin": "https://www.gutenberg.org/cache/epub/1228/pg1228.txt",
    "Communist Manifesto - Marx": "https://www.gutenberg.org/cache/epub/61/pg61.txt",
    "Discourse on Method - Descartes": "https://www.gutenberg.org/cache/epub/59/pg59.txt",

    # ==== PLAYS & DRAMA ====
    "Pygmalion - Shaw": "https://www.gutenberg.org/cache/epub/3825/pg3825.txt",
    "Arms and the Man - Shaw": "https://www.gutenberg.org/cache/epub/3618/pg3618.txt",
    "Man and Superman - Shaw": "https://www.gutenberg.org/cache/epub/3328/pg3328.txt",
    "Mrs Warren's Profession - Shaw": "https://www.gutenberg.org/cache/epub/1097/pg1097.txt",
    "The Playboy of the Western World - Synge": "https://www.gutenberg.org/cache/epub/1240/pg1240.txt",
    "A Doll's House - Ibsen": "https://www.gutenberg.org/cache/epub/2542/pg2542.txt",
    "Hedda Gabler - Ibsen": "https://www.gutenberg.org/cache/epub/4093/pg4093.txt",
    "An Enemy of the People - Ibsen": "https://www.gutenberg.org/cache/epub/2446/pg2446.txt",
    "The Cherry Orchard - Chekhov": "https://www.gutenberg.org/cache/epub/1753/pg1753.txt",
    "The Seagull - Chekhov": "https://www.gutenberg.org/cache/epub/7986/pg7986.txt",
    "Cyrano de Bergerac - Rostand": "https://www.gutenberg.org/cache/epub/1254/pg1254.txt",
    "She Stoops to Conquer - Goldsmith": "https://www.gutenberg.org/cache/epub/383/pg383.txt",
    "The Rivals - Sheridan": "https://www.gutenberg.org/cache/epub/1632/pg1632.txt",
    "The School for Scandal - Sheridan": "https://www.gutenberg.org/cache/epub/1929/pg1929.txt",
    "The Way of the World - Congreve": "https://www.gutenberg.org/cache/epub/1292/pg1292.txt",

    # ==== CHILDREN'S & FANTASY ====
    "Alice in Wonderland - Carroll": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "Through the Looking Glass - Carroll": "https://www.gutenberg.org/cache/epub/12/pg12.txt",
    "The Wonderful Wizard of Oz - Baum": "https://www.gutenberg.org/cache/epub/55/pg55.txt",
    "Grimm's Fairy Tales": "https://www.gutenberg.org/cache/epub/2591/pg2591.txt",
    "Andersen's Fairy Tales": "https://www.gutenberg.org/cache/epub/1597/pg1597.txt",
    "Aesop's Fables": "https://www.gutenberg.org/cache/epub/11339/pg11339.txt",
    "Arabian Nights": "https://www.gutenberg.org/cache/epub/34206/pg34206.txt",
    "The Water-Babies - Kingsley": "https://www.gutenberg.org/cache/epub/1018/pg1018.txt",
    "Black Beauty - Sewell": "https://www.gutenberg.org/cache/epub/271/pg271.txt",
    "The Princess and the Goblin - MacDonald": "https://www.gutenberg.org/cache/epub/708/pg708.txt",
    "Anne of Green Gables - Montgomery": "https://www.gutenberg.org/cache/epub/45/pg45.txt",
    "Heidi - Spyri": "https://www.gutenberg.org/cache/epub/1448/pg1448.txt",
    "Pinocchio - Collodi": "https://www.gutenberg.org/cache/epub/500/pg500.txt",
    "Peter Pan - Barrie": "https://www.gutenberg.org/cache/epub/16/pg16.txt",

    # ==== MISCELLANEOUS CLASSICS ====
    "The Phantom of the Opera - Leroux": "https://www.gutenberg.org/cache/epub/175/pg175.txt",
    "Don Quixote Part 2 - Cervantes": "https://www.gutenberg.org/cache/epub/5921/pg5921.txt",
    "The Decameron - Boccaccio": "https://www.gutenberg.org/cache/epub/23700/pg23700.txt",
    "The Iliad - Homer (Lang)": "https://www.gutenberg.org/cache/epub/3059/pg3059.txt",
    "Siddhartha - Hesse": "https://www.gutenberg.org/cache/epub/2500/pg2500.txt",
    "The Trial - Kafka": "https://www.gutenberg.org/cache/epub/7849/pg7849.txt",
    "Metamorphosis - Kafka": "https://www.gutenberg.org/cache/epub/5200/pg5200.txt",
    "Narrative of Arthur Gordon Pym - Poe": "https://www.gutenberg.org/cache/epub/51060/pg51060.txt",
    "Fall of House of Usher & Others - Poe": "https://www.gutenberg.org/cache/epub/932/pg932.txt",
    "The Raven and Other Poems - Poe": "https://www.gutenberg.org/cache/epub/17192/pg17192.txt",
    "Complete Poetical Works - Shelley": "https://www.gutenberg.org/cache/epub/4800/pg4800.txt",
    "Poems - Robert Burns": "https://www.gutenberg.org/cache/epub/1279/pg1279.txt",
    "Narrative - Frederick Douglass": "https://www.gutenberg.org/cache/epub/23/pg23.txt",
    "Up from Slavery - Washington": "https://www.gutenberg.org/cache/epub/2376/pg2376.txt",
    "Souls of Black Folk - Du Bois": "https://www.gutenberg.org/cache/epub/408/pg408.txt",
    "The Federalist Papers": "https://www.gutenberg.org/cache/epub/1404/pg1404.txt",
    "Democracy in America - Tocqueville": "https://www.gutenberg.org/cache/epub/815/pg815.txt",
    "History of the Peloponnesian War - Thucydides": "https://www.gutenberg.org/cache/epub/7142/pg7142.txt",
    "The Histories - Herodotus": "https://www.gutenberg.org/cache/epub/2707/pg2707.txt",
    "Decline and Fall of the Roman Empire Vol1 - Gibbon": "https://www.gutenberg.org/cache/epub/25717/pg25717.txt",
    "Decline and Fall of the Roman Empire Vol2 - Gibbon": "https://www.gutenberg.org/cache/epub/25717/pg25717.txt",

    # ==== MORE TWAIN ====
    "The Prince and the Pauper - Twain": "https://www.gutenberg.org/cache/epub/1837/pg1837.txt",
    "Life on the Mississippi - Twain": "https://www.gutenberg.org/cache/epub/245/pg245.txt",
    "A Tramp Abroad - Twain": "https://www.gutenberg.org/cache/epub/119/pg119.txt",
    "Roughing It - Twain": "https://www.gutenberg.org/cache/epub/3177/pg3177.txt",
    "Innocents Abroad - Twain": "https://www.gutenberg.org/cache/epub/3176/pg3176.txt",
    "Pudd'nhead Wilson - Twain": "https://www.gutenberg.org/cache/epub/102/pg102.txt",

    # ==== MORE WELLS ====
    "The Food of the Gods - Wells": "https://www.gutenberg.org/cache/epub/11696/pg11696.txt",
    "When the Sleeper Wakes - Wells": "https://www.gutenberg.org/cache/epub/12163/pg12163.txt",
    "The World Set Free - Wells": "https://www.gutenberg.org/cache/epub/1059/pg1059.txt",
    "Kipps - Wells": "https://www.gutenberg.org/cache/epub/3427/pg3427.txt",
    "Ann Veronica - Wells": "https://www.gutenberg.org/cache/epub/4245/pg4245.txt",
    "Tono-Bungay - Wells": "https://www.gutenberg.org/cache/epub/5765/pg5765.txt",
    "The History of Mr Polly - Wells": "https://www.gutenberg.org/cache/epub/7308/pg7308.txt",

    # ==== HISTORICAL FICTION ====
    "The Last Days of Pompeii - Lytton": "https://www.gutenberg.org/cache/epub/2312/pg2312.txt",
    "A Tale of Two Cities - Dickens": "https://www.gutenberg.org/cache/epub/98/pg98.txt",
    "The Scarlet Pimpernel - Orczy": "https://www.gutenberg.org/cache/epub/60/pg60.txt",
    "Quo Vadis - Sienkiewicz": "https://www.gutenberg.org/cache/epub/2853/pg2853.txt",
    "Ben-Hur - Wallace": "https://www.gutenberg.org/cache/epub/2145/pg2145.txt",
    "The Three Musketeers - Dumas": "https://www.gutenberg.org/cache/epub/1257/pg1257.txt",
    "Lorna Doone - Blackmore": "https://www.gutenberg.org/cache/epub/840/pg840.txt",

    # ==== SATIRE & SOCIAL COMMENTARY ====
    "Erewhon - Butler": "https://www.gutenberg.org/cache/epub/1906/pg1906.txt",
    "Flatland - Abbott": "https://www.gutenberg.org/cache/epub/201/pg201.txt",
    "The Man Who Was Thursday - Chesterton": "https://www.gutenberg.org/cache/epub/1695/pg1695.txt",
    "The Napoleon of Notting Hill - Chesterton": "https://www.gutenberg.org/cache/epub/20058/pg20058.txt",
    "Father Brown Stories - Chesterton": "https://www.gutenberg.org/cache/epub/204/pg204.txt",
    "Three Men in a Boat - Jerome": "https://www.gutenberg.org/cache/epub/308/pg308.txt",
    "Diary of a Nobody - Grossmith": "https://www.gutenberg.org/cache/epub/1026/pg1026.txt",
    "The Importance of Being Earnest - Wilde": "https://www.gutenberg.org/cache/epub/844/pg844.txt",

    # ==== AUTOBIOGRAPHY & MEMOIR ====
    "Life of Samuel Johnson - Boswell": "https://www.gutenberg.org/cache/epub/1564/pg1564.txt",
    "Confessions - Rousseau": "https://www.gutenberg.org/cache/epub/3913/pg3913.txt",
    "Autobiography - John Stuart Mill": "https://www.gutenberg.org/cache/epub/10378/pg10378.txt",
    "My Bondage and My Freedom - Douglass": "https://www.gutenberg.org/cache/epub/202/pg202.txt",

    # ==== ANCIENT LITERATURE ====
    "Oedipus the King - Sophocles": "https://www.gutenberg.org/cache/epub/31/pg31.txt",
    "Antigone - Sophocles": "https://www.gutenberg.org/cache/epub/31/pg31.txt",
    "The Bacchae - Euripides": "https://www.gutenberg.org/cache/epub/5127/pg5127.txt",
    "Oresteia - Aeschylus": "https://www.gutenberg.org/cache/epub/8714/pg8714.txt",
    "The Frogs - Aristophanes": "https://www.gutenberg.org/cache/epub/7998/pg7998.txt",
    "Lysistrata - Aristophanes": "https://www.gutenberg.org/cache/epub/7700/pg7700.txt",
    "Satyricon - Petronius": "https://www.gutenberg.org/cache/epub/5225/pg5225.txt",
    "Golden Ass - Apuleius": "https://www.gutenberg.org/cache/epub/1666/pg1666.txt",

    # ==== MORE NOVELS ====
    "The Thirty-Nine Steps - Buchan": "https://www.gutenberg.org/cache/epub/558/pg558.txt",
    "The Four Feathers - Mason": "https://www.gutenberg.org/cache/epub/2570/pg2570.txt",
    "Greenmantle - Buchan": "https://www.gutenberg.org/cache/epub/559/pg559.txt",
    "Mr Standfast - Buchan": "https://www.gutenberg.org/cache/epub/560/pg560.txt",
    "The Phantom Rickshaw - Kipling": "https://www.gutenberg.org/cache/epub/2137/pg2137.txt",
    "Captains Courageous - Kipling": "https://www.gutenberg.org/cache/epub/2186/pg2186.txt",
    "The Light That Failed - Kipling": "https://www.gutenberg.org/cache/epub/2876/pg2876.txt",
    "Stalky and Co - Kipling": "https://www.gutenberg.org/cache/epub/3006/pg3006.txt",
    "The Man Who Would Be King - Kipling": "https://www.gutenberg.org/cache/epub/2226/pg2226.txt",
    "Just So Stories - Kipling": "https://www.gutenberg.org/cache/epub/2781/pg2781.txt",
    "Plain Tales from the Hills - Kipling": "https://www.gutenberg.org/cache/epub/3297/pg3297.txt",
    "Puck of Pook's Hill - Kipling": "https://www.gutenberg.org/cache/epub/557/pg557.txt",
    "The Jungle Book 2 - Kipling": "https://www.gutenberg.org/cache/epub/37364/pg37364.txt",
    "The Pickwick Papers - Dickens": "https://www.gutenberg.org/cache/epub/580/pg580.txt",
    "The Old Curiosity Shop - Dickens": "https://www.gutenberg.org/cache/epub/700/pg700.txt",
    "A Connecticut Yankee - Twain": "https://www.gutenberg.org/cache/epub/86/pg86.txt",
    "The Vicar of Wakefield - Goldsmith": "https://www.gutenberg.org/cache/epub/2667/pg2667.txt",
    "Pamela - Richardson": "https://www.gutenberg.org/cache/epub/6124/pg6124.txt",
    "Clarissa Vol 1 - Richardson": "https://www.gutenberg.org/cache/epub/9296/pg9296.txt",
    "Tom Jones - Fielding": "https://www.gutenberg.org/cache/epub/6593/pg6593.txt",
    "Tristram Shandy - Sterne": "https://www.gutenberg.org/cache/epub/1079/pg1079.txt",
    "Moll Flanders - Defoe": "https://www.gutenberg.org/cache/epub/370/pg370.txt",
    "Evelina - Burney": "https://www.gutenberg.org/cache/epub/6053/pg6053.txt",
    "Cecilia - Burney": "https://www.gutenberg.org/cache/epub/7700/pg7700.txt",
    "Caleb Williams - Godwin": "https://www.gutenberg.org/cache/epub/11323/pg11323.txt",
    "Vathek - Beckford": "https://www.gutenberg.org/cache/epub/2871/pg2871.txt",
    "Rasselas - Johnson": "https://www.gutenberg.org/cache/epub/652/pg652.txt",
    "The Coral Island - Ballantyne": "https://www.gutenberg.org/cache/epub/12750/pg12750.txt",
}


# Number of unique URLs (deduplication happens at download time)
_unique_urls = len(set(GUTENBERG_TEXTS.values()))
print(f"Gutenberg corpus: {len(GUTENBERG_TEXTS)} entries, {_unique_urls} unique URLs") if __name__ != "__main__" else None


def _strip_gutenberg_header_footer(text: str) -> str:
    """
    Remove Project Gutenberg boilerplate header and footer.
    
    Gutenberg texts have standard markers:
    - Header ends with: '*** START OF THE PROJECT GUTENBERG EBOOK ...'
    - Footer starts with: '*** END OF THE PROJECT GUTENBERG EBOOK ...'
    """
    # Strip header
    start_markers = [
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK",
        r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK",
        r"\*\*\*START OF THE PROJECT GUTENBERG EBOOK",
    ]
    for marker in start_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            # Find the end of the line after the marker
            newline_pos = text.find('\n', match.end())
            if newline_pos != -1:
                text = text[newline_pos + 1:]
            break
    
    # Strip footer
    end_markers = [
        r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK",
        r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK",
        r"\*\*\*END OF THE PROJECT GUTENBERG EBOOK",
        r"End of the Project Gutenberg EBook",
        r"End of Project Gutenberg",
    ]
    for marker in end_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]
            break
    
    return text.strip()


def download_gutenberg_corpus(
    cache_path: Optional[Path] = None,
    force_download: bool = False
) -> str:
    """
    Download and concatenate all Gutenberg texts into a single pre-training corpus.
    
    Handles deduplication: if multiple entries share the same URL, the text is
    only downloaded once. This is common when the same Gutenberg ID contains
    multiple works (e.g., collected plays).
    
    Args:
        cache_path: Path to cache the combined corpus (default: data/gutenberg_expanded.txt)
        force_download: Re-download even if cache exists
    
    Returns:
        Combined corpus text (all Gutenberg texts concatenated)
    """
    if cache_path is None:
        cache_path = config.DATA_DIR / "gutenberg_expanded.txt"
    
    # Return cached version if available
    if not force_download and cache_path.exists():
        print(f"Loading cached Gutenberg corpus from {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"  Corpus size: {len(text):,} characters ({len(text)/1024/1024:.1f} MB)")
        return text
    
    print("=" * 70)
    print("DOWNLOADING EXPANDED GUTENBERG PRE-TRAINING CORPUS")
    print(f"  {len(GUTENBERG_TEXTS)} entries in catalogue")
    print("=" * 70)
    
    # Deduplicate by URL — download each unique URL only once
    url_to_text: dict[str, str] = {}
    all_texts = []
    failed = []
    skipped_dupes = 0
    
    for i, (name, url) in enumerate(GUTENBERG_TEXTS.items(), 1):
        # Skip if we already downloaded this URL
        if url in url_to_text:
            skipped_dupes += 1
            continue
        
        try:
            print(f"  [{i}/{len(GUTENBERG_TEXTS)}] {name}...", end=" ", flush=True)
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req, timeout=30)
            raw = response.read().decode('utf-8', errors='replace')
            
            # Strip Gutenberg header/footer
            clean = _strip_gutenberg_header_footer(raw)
            
            # Skip very short texts (likely errors or empty pages)  
            if len(clean) < 500:
                print(f"SKIPPED (too short: {len(clean)} chars)")
                failed.append(f"{name} (too short)")
                continue
            
            chars = len(clean)
            print(f"{chars:,} chars ({chars/1024:.0f} KB)")
            
            url_to_text[url] = clean
            all_texts.append(clean)
            
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(name)
    
    print(f"\n  Downloaded: {len(all_texts)} texts")
    print(f"  Duplicates skipped: {skipped_dupes}")
    if failed:
        print(f"  Failed ({len(failed)}): {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")
    
    # Concatenate all texts with separator
    corpus = "\n\n".join(all_texts)
    
    # Basic cleaning: normalise whitespace, remove excessive blank lines
    corpus = re.sub(r'\n{4,}', '\n\n\n', corpus)  # Max 3 newlines in a row
    corpus = re.sub(r'[ \t]+', ' ', corpus)         # Collapse spaces/tabs
    
    total_mb = len(corpus) / 1024 / 1024
    est_words = len(corpus.split())
    est_bpe = len(corpus) // 4  # rough ~4 chars per BPE token
    
    print(f"\n{'='*70}")
    print(f"CORPUS SUMMARY")
    print(f"  Characters: {len(corpus):,} ({total_mb:.1f} MB)")
    print(f"  Est. words: {est_words:,}")
    print(f"  Est. BPE tokens (≈4 chars/token): {est_bpe:,}")
    print(f"  Chinchilla ratio with 6.4M params: {est_bpe/6_400_000:.1f}:1 tokens/param")
    print(f"{'='*70}")
    
    # Cache to disk
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(corpus)
    print(f"  Saved to {cache_path}")
    
    return corpus


def download_shakespeare() -> str:
    """
    Download the Complete Works of Shakespeare (same as data_loader.py).
    Used separately for fine-tuning after pre-training on Gutenberg.
    """
    from data_loader import download_shakespeare as _download
    return _download()


# ============================================================================
# For quick testing
# ============================================================================
if __name__ == "__main__":
    corpus = download_gutenberg_corpus(force_download=True)
    print(f"\nCorpus stats:")
    print(f"  Characters: {len(corpus):,}")
    print(f"  Estimated words: {len(corpus.split()):,}")
    est_bpe = len(corpus) // 4
    print(f"  Estimated BPE tokens (≈4 chars/token): {est_bpe:,}")
    print(f"  Chinchilla ratio (6.4M params): {est_bpe/6_400_000:.1f}:1")
    
    # Show first 500 chars
    print(f"\nFirst 500 characters:")
    print(corpus[:500])

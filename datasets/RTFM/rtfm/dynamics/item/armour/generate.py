# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python
"""
This file will generate the weapons from the Nethack wiki at
https://nethackwiki.com/wiki/Armor
"""
import os
import re
from lxml import html


TABLE = """
<table class="prettytable">
<tbody><tr>
<th>Name
</th>
<th>Cost
</th>
<th>Weight
</th>
<th><a href="/wiki/Armor_class" title="Armor class">AC</a>
</th>
<th>Weight per AC (+0)
</th>
<th>Weight per AC (+5)
</th>
<th>Material
</th>
<th>Effect
</th>
<th><a href="/wiki/Magic_cancellation" title="Magic cancellation">MC</a>
</th>
<th>Prob<sup id="cite_ref-6" class="reference"><a href="#cite_note-6">[6]</a></sup> (‰)
</th>
<th><a href="/wiki/Magical_object" class="mw-redirect" title="Magical object">Magical</a>
</th>
<th>Appearance
</th></tr>
<tr>
<td colspan="10"><b><a href="/wiki/Shirt" title="Shirt">Shirts</a></b>
</td></tr>
<tr>
<td><a href="/wiki/Hawaiian_shirt" class="mw-redirect" title="Hawaiian shirt">Hawaiian shirt</a></td>
<td>3</td>
<td>5</td>
<td>0</td>
<td>Infinite</td>
<td>1</td>
<td>cloth</td>
<td>Shop</td>
<td></td>
<td>8</td>
<td></td>
<td>--
</td></tr>
<tr>
<td><a href="/wiki/T-shirt" class="mw-redirect" title="T-shirt">T-shirt</a></td>
<td>2</td>
<td>5</td>
<td>0</td>
<td>Infinite</td>
<td>1</td>
<td>cloth</td>
<td>Shop</td>
<td></td>
<td>2</td>
<td></td>
<td>--
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Body_armor" title="Body armor"><b>Suits</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Leather_jacket" title="Leather jacket">leather jacket</a></td>
<td>10</td>
<td>30</td>
<td>1</td>
<td>30</td>
<td>5</td>
<td>leather</td>
<td></td>
<td></td>
<td>12</td>
<td></td>
<td>--
</td></tr>
<tr>
<td><a href="/wiki/Leather_armor" title="Leather armor">leather armor</a></td>
<td>5</td>
<td>150</td>
<td>2</td>
<td>75</td>
<td>21</td>
<td>leather</td>
<td></td>
<td>1</td>
<td>82</td>
<td></td>
<td>--
</td></tr>
<tr>
<td><a href="/wiki/Orcish_ring_mail" title="Orcish ring mail">orcish ring mail</a></td>
<td>80</td>
<td>250</td>
<td>2</td>
<td>125</td>
<td>36</td>
<td>iron</td>
<td></td>
<td>1</td>
<td>20</td>
<td></td>
<td>crude ring mail
</td></tr>
<tr>
<td><a href="/wiki/Studded_leather_armor" title="Studded leather armor">studded leather armor</a></td>
<td>15</td>
<td>200</td>
<td>3</td>
<td>67</td>
<td>25</td>
<td>leather</td>
<td></td>
<td>1</td>
<td>72</td>
<td></td>
<td>--
</td></tr>
<tr>
<td><a href="/wiki/Ring_mail" title="Ring mail">ring mail</a></td>
<td>100</td>
<td>250</td>
<td>3</td>
<td>83</td>
<td></td>
<td>iron</td>
<td></td>
<td>1</td>
<td>72</td>
<td></td>
<td>--
</td></tr>
<tr>
<td><a href="/wiki/Scale_mail" title="Scale mail">scale mail</a></td>
<td>45</td>
<td>250</td>
<td>4</td>
<td>63</td>
<td></td>
<td>iron</td>
<td></td>
<td>1</td>
<td>72</td>
<td></td>
<td>--
</td></tr>
<tr>
<td><a href="/wiki/Orcish_chain_mail" title="Orcish chain mail">orcish chain mail</a></td>
<td>75</td>
<td>300</td>
<td>4</td>
<td>75</td>
<td></td>
<td>iron</td>
<td></td>
<td>1</td>
<td>20</td>
<td></td>
<td>crude chain mail
</td></tr>
<tr>
<td><a href="/wiki/Chain_mail" title="Chain mail">chain mail</a></td>
<td>75</td>
<td>300</td>
<td>5</td>
<td>60</td>
<td></td>
<td>iron</td>
<td></td>
<td>1</td>
<td>72</td>
<td></td>
<td>--
</td></tr>
<tr>
<td><a href="/wiki/Elven_mithril-coat" class="mw-redirect" title="Elven mithril-coat">elven mithril-coat</a></td>
<td>240</td>
<td>150</td>
<td>5</td>
<td>30</td>
<td></td>
<td>mithril</td>
<td></td>
<td>2</td>
<td>15</td>
<td></td>
<td>--
</td></tr>
<tr>
<td><a href="/wiki/Splint_mail" title="Splint mail">splint mail</a></td>
<td>80</td>
<td>400</td>
<td>6</td>
<td>67</td>
<td></td>
<td>iron</td>
<td></td>
<td>1</td>
<td>62</td>
<td></td>
<td>--
</td></tr>
<tr>
<td><a href="/wiki/Banded_mail" title="Banded mail">banded mail</a></td>
<td>90</td>
<td>350</td>
<td>6</td>
<td>58</td>
<td></td>
<td>iron</td>
<td></td>
<td>1</td>
<td>72</td>
<td></td>
<td>--
</td></tr>
<tr>
<td><a href="/wiki/Dwarvish_mithril-coat" class="mw-redirect" title="Dwarvish mithril-coat">dwarvish mithril-coat</a></td>
<td>240</td>
<td>150</td>
<td>6</td>
<td>25</td>
<td></td>
<td>mithril</td>
<td></td>
<td>2</td>
<td>10</td>
<td></td>
<td>--
</td></tr>
<tr>
<td><a href="/wiki/Bronze_plate_mail" title="Bronze plate mail">bronze plate mail</a></td>
<td>400</td>
<td>450</td>
<td>6</td>
<td>75</td>
<td></td>
<td>copper</td>
<td></td>
<td>1</td>
<td>25</td>
<td></td>
<td>--
</td></tr>
<tr>
<td><a href="/wiki/Plate_mail" title="Plate mail">plate mail (tanko)</a></td>
<td>600</td>
<td>450</td>
<td>7</td>
<td>64</td>
<td></td>
<td>iron</td>
<td></td>
<td>2</td>
<td>44</td>
<td></td>
<td>--
</td></tr>
<tr>
<td><a href="/wiki/Crystal_plate_mail" title="Crystal plate mail">crystal plate mail</a></td>
<td>820</td>
<td>450</td>
<td>7</td>
<td>64</td>
<td></td>
<td>glass</td>
<td></td>
<td>2</td>
<td>10</td>
<td></td>
<td>--
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Dragon_scale_mail" title="Dragon scale mail"><b>Dragon suits</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Dragon_scales" class="mw-redirect" title="Dragon scales">dragon scales</a></td>
<td>500</td>
<td>40</td>
<td>3</td>
<td>13</td>
<td></td>
<td>dragon</td>
<td>Resist<sup>%</sup></td>
<td></td>
<td>--</td>
<td></td>
<td>--
</td></tr>
<tr>
<td><a href="/wiki/Dragon_scale_mail" title="Dragon scale mail">dragon scale mail</a></td>
<td>900</td>
<td>40</td>
<td>9</td>
<td>4</td>
<td></td>
<td>dragon</td>
<td>Resist<sup>%</sup></td>
<td></td>
<td>--</td>
<td>Yes</td>
<td>--
</td></tr>
<tr>
<td colspan="10"><b><a href="/wiki/Cloak" title="Cloak">Cloaks</a></b>
</td></tr>
<tr>
<td><a href="/wiki/Mummy_wrapping" title="Mummy wrapping">mummy wrapping</a></td>
<td>2</td>
<td>3</td>
<td>0</td>
<td>Infinite</td>
<td></td>
<td>cloth</td>
<td>Vis</td>
<td>1</td>
<td>--</td>
<td></td>
<td>--
</td></tr>
<tr>
<td><a href="/wiki/Orcish_cloak" title="Orcish cloak">orcish cloak</a></td>
<td>40</td>
<td>10</td>
<td>0</td>
<td>Infinite</td>
<td></td>
<td>cloth</td>
<td></td>
<td>1</td>
<td>8</td>
<td></td>
<td>coarse mantelet
</td></tr>
<tr>
<td><a href="/wiki/Dwarvish_cloak" title="Dwarvish cloak">dwarvish cloak</a></td>
<td>50</td>
<td>10</td>
<td>0</td>
<td>Infinite</td>
<td></td>
<td>cloth</td>
<td></td>
<td>1</td>
<td>8</td>
<td></td>
<td>hooded cloak
</td></tr>
<tr>
<td><a href="/wiki/Leather_cloak" title="Leather cloak">leather cloak</a></td>
<td>40</td>
<td>15</td>
<td>1</td>
<td>15</td>
<td></td>
<td>leather</td>
<td></td>
<td>1</td>
<td>8</td>
<td></td>
<td>--
</td></tr>
<tr>
<td><a href="/wiki/Cloak_of_displacement" title="Cloak of displacement">cloak of displacement</a></td>
<td>50</td>
<td>10</td>
<td>1</td>
<td>10</td>
<td></td>
<td>cloth</td>
<td>Displ</td>
<td>1</td>
<td>10</td>
<td>Yes</td>
<td>*piece of cloth
</td></tr>
<tr>
<td><a href="/wiki/Oilskin_cloak" title="Oilskin cloak">oilskin cloak</a></td>
<td>50</td>
<td>10</td>
<td>1</td>
<td>10</td>
<td></td>
<td>cloth</td>
<td>Water</td>
<td>2</td>
<td>8</td>
<td></td>
<td>slippery cloak
</td></tr>
<tr>
<td><a href="/wiki/Alchemy_smock" title="Alchemy smock">alchemy smock</a></td>
<td>50</td>
<td>10</td>
<td>1</td>
<td>10</td>
<td></td>
<td>cloth</td>
<td>Poi+Acd</td>
<td>1</td>
<td>9</td>
<td>Yes</td>
<td>apron
</td></tr>
<tr>
<td><a href="/wiki/Cloak_of_invisibility" title="Cloak of invisibility">cloak of invisibility</a></td>
<td>60</td>
<td>10</td>
<td>1</td>
<td>10</td>
<td></td>
<td>cloth</td>
<td>Invis</td>
<td>1</td>
<td>10</td>
<td>Yes</td>
<td>*opera cloak
</td></tr>
<tr>
<td><a href="/wiki/Cloak_of_magic_resistance" title="Cloak of magic resistance">cloak of magic resistance</a></td>
<td>60</td>
<td>10</td>
<td>1</td>
<td>10</td>
<td></td>
<td>cloth</td>
<td>Magic</td>
<td>1</td>
<td>2</td>
<td>Yes</td>
<td>*ornamental cope
</td></tr>
<tr>
<td><a href="/wiki/Elven_cloak" title="Elven cloak">elven cloak </a></td>
<td>60</td>
<td>10</td>
<td>1</td>
<td>10</td>
<td></td>
<td>cloth</td>
<td>Stealth</td>
<td>1</td>
<td>8</td>
<td>Yes</td>
<td>faded pall
</td></tr>
<tr>
<td><a href="/wiki/Robe" title="Robe">robe</a></td>
<td>50</td>
<td>15</td>
<td>2</td>
<td>8</td>
<td></td>
<td>cloth</td>
<td>Spell</td>
<td>2</td>
<td>3</td>
<td>Yes</td>
<td>--
</td></tr>
<tr>
<td><a href="/wiki/Cloak_of_protection" title="Cloak of protection">cloak of protection</a></td>
<td>50</td>
<td>10</td>
<td>3</td>
<td>3</td>
<td></td>
<td>cloth</td>
<td>Prot</td>
<td>3</td>
<td>9</td>
<td>Yes</td>
<td>*tattered cape
</td></tr>
<tr>
<td colspan="10"><b><a href="/wiki/Helm" title="Helm">Helms</a></b>
</td></tr>
<tr>
<td><a href="/wiki/Fedora" title="Fedora">fedora</a></td>
<td>1</td>
<td>3</td>
<td>0</td>
<td>Infinite</td>
<td></td>
<td>cloth</td>
<td></td>
<td></td>
<td>--</td>
<td></td>
<td>--
</td></tr>
<tr>
<td><a href="/wiki/Dunce_cap" title="Dunce cap">dunce cap</a></td>
<td>1</td>
<td>4</td>
<td>0</td>
<td>Infinite</td>
<td></td>
<td>cloth</td>
<td>Stupid</td>
<td></td>
<td>3</td>
<td>Yes</td>
<td>conical hat
</td></tr>
<tr>
<td><a href="/wiki/Cornuthaum" title="Cornuthaum">cornuthaum</a></td>
<td>80</td>
<td>4</td>
<td>0</td>
<td>Infinite</td>
<td></td>
<td>cloth</td>
<td>Clair</td>
<td>1</td>
<td>3</td>
<td>Yes</td>
<td>conical hat
</td></tr>
<tr>
<td><a href="/wiki/Dented_pot" title="Dented pot">dented pot</a></td>
<td>8</td>
<td>10</td>
<td>1</td>
<td>10</td>
<td></td>
<td>iron</td>
<td></td>
<td></td>
<td>2</td>
<td></td>
<td>--
</td></tr>
<tr>
<td><a href="/wiki/Elven_leather_helm" title="Elven leather helm">elven leather helm</a></td>
<td>8</td>
<td>3</td>
<td>1</td>
<td>3</td>
<td></td>
<td>leather</td>
<td></td>
<td></td>
<td>6</td>
<td></td>
<td>leather hat
</td></tr>
<tr>
<td><a href="/wiki/Helmet" title="Helmet">helmet (kabuto)</a></td>
<td>10</td>
<td>30</td>
<td>1</td>
<td>30</td>
<td></td>
<td>iron</td>
<td></td>
<td></td>
<td>10</td>
<td></td>
<td>*plumed helmet
</td></tr>
<tr>
<td><a href="/wiki/Orcish_helm" title="Orcish helm">orcish helm</a></td>
<td>10</td>
<td>30</td>
<td>1</td>
<td>30</td>
<td></td>
<td>iron</td>
<td></td>
<td></td>
<td>6</td>
<td></td>
<td>iron skull cap
</td></tr>
<tr>
<td><a href="/wiki/Helm_of_brilliance" title="Helm of brilliance">helm of brilliance</a></td>
<td>50</td>
<td>50</td>
<td>1</td>
<td>50</td>
<td></td>
<td>iron</td>
<td>Int+Wis</td>
<td></td>
<td>6</td>
<td>Yes</td>
<td>*etched helmet
</td></tr>
<tr>
<td><a href="/wiki/Helm_of_opposite_alignment" title="Helm of opposite alignment">helm of opposite alignment</a></td>
<td>50</td>
<td>50</td>
<td>1</td>
<td>50</td>
<td></td>
<td>iron</td>
<td>Align</td>
<td></td>
<td>6</td>
<td>Yes</td>
<td>*crested helmet
</td></tr>
<tr>
<td><a href="/wiki/Helm_of_telepathy" title="Helm of telepathy">helm of telepathy</a></td>
<td>50</td>
<td>50</td>
<td>1</td>
<td>50</td>
<td></td>
<td>iron</td>
<td>ESP</td>
<td></td>
<td>2</td>
<td>Yes</td>
<td>*visored helmet
</td></tr>
<tr>
<td><a href="/wiki/Dwarvish_iron_helm" title="Dwarvish iron helm">dwarvish iron helm</a></td>
<td>20</td>
<td>40</td>
<td>2</td>
<td>20</td>
<td></td>
<td>iron</td>
<td></td>
<td></td>
<td>6</td>
<td></td>
<td>hard hat
</td></tr>
<tr>
<td colspan="10"><b><a href="/wiki/Gloves" title="Gloves">Gloves</a></b>
</td></tr>
<tr>
<td><a href="/wiki/Leather_gloves" title="Leather gloves">leather gloves (yugake)</a></td>
<td>8</td>
<td>10</td>
<td>1</td>
<td>10</td>
<td></td>
<td>leather</td>
<td></td>
<td></td>
<td>16</td>
<td></td>
<td>*old gloves
</td></tr>
<tr>
<td><a href="/wiki/Gauntlets_of_dexterity" title="Gauntlets of dexterity">gauntlets of dexterity</a></td>
<td>50</td>
<td>10</td>
<td>1</td>
<td>10</td>
<td></td>
<td>leather</td>
<td>Dex</td>
<td></td>
<td>8</td>
<td>Yes</td>
<td>*padded gloves
</td></tr>
<tr>
<td><a href="/wiki/Gauntlets_of_fumbling" title="Gauntlets of fumbling">gauntlets of fumbling</a></td>
<td>50</td>
<td>10</td>
<td>1</td>
<td>10</td>
<td></td>
<td>leather</td>
<td>Fumble</td>
<td></td>
<td>8</td>
<td>Yes</td>
<td>*riding gloves
</td></tr>
<tr>
<td><a href="/wiki/Gauntlets_of_power" title="Gauntlets of power">gauntlets of power</a></td>
<td>50</td>
<td>30</td>
<td>1</td>
<td>30</td>
<td></td>
<td>iron</td>
<td>Str</td>
<td></td>
<td>8</td>
<td>Yes</td>
<td>*fencing gloves
</td></tr>
<tr>
<td colspan="10"><b><a href="/wiki/Shield" title="Shield">Shields</a></b>
</td></tr>
<tr>
<td><a href="/wiki/Small_shield" title="Small shield">small shield</a></td>
<td>3</td>
<td>30</td>
<td>1</td>
<td>30</td>
<td></td>
<td>wood</td>
<td></td>
<td></td>
<td>6</td>
<td></td>
<td>--
</td></tr>
<tr>
<td><a href="/wiki/Orcish_shield" title="Orcish shield">orcish shield</a></td>
<td>7</td>
<td>50</td>
<td>1</td>
<td>50</td>
<td></td>
<td>iron</td>
<td></td>
<td></td>
<td>2</td>
<td></td>
<td>red-eyed
</td></tr>
<tr>
<td><a href="/wiki/Uruk-hai_shield" title="Uruk-hai shield">Uruk-hai shield</a></td>
<td>7</td>
<td>50</td>
<td>1</td>
<td>50</td>
<td></td>
<td>iron</td>
<td></td>
<td></td>
<td>2</td>
<td></td>
<td>white-handed
</td></tr>
<tr>
<td><a href="/wiki/Elven_shield" title="Elven shield">elven shield</a></td>
<td>7</td>
<td>40</td>
<td>2</td>
<td>20</td>
<td></td>
<td>wood</td>
<td></td>
<td></td>
<td>2</td>
<td></td>
<td>blue and green
</td></tr>
<tr>
<td><a href="/wiki/Dwarvish_roundshield" title="Dwarvish roundshield">dwarvish roundshield</a></td>
<td>10</td>
<td>100</td>
<td>2</td>
<td>50</td>
<td></td>
<td>iron</td>
<td></td>
<td></td>
<td>4</td>
<td></td>
<td>large round
</td></tr>
<tr>
<td><a href="/wiki/Large_shield" title="Large shield">large shield</a></td>
<td>10</td>
<td>100</td>
<td>2</td>
<td>50</td>
<td></td>
<td>iron</td>
<td></td>
<td></td>
<td>7</td>
<td></td>
<td>--
</td></tr>
<tr>
<td><a href="/wiki/Shield_of_reflection" title="Shield of reflection">shield of reflection</a></td>
<td>50</td>
<td>50</td>
<td>2</td>
<td>25</td>
<td></td>
<td>silver</td>
<td>Reflect</td>
<td></td>
<td>3</td>
<td>Yes</td>
<td>polished silver
</td></tr>
<tr>
<td colspan="10"><b><a href="/wiki/Boots" title="Boots">Boots</a></b>
</td></tr>
<tr>
<td><a href="/wiki/Low_boots" title="Low boots">low boots</a></td>
<td>8</td>
<td>10</td>
<td>1</td>
<td>10</td>
<td></td>
<td>leather</td>
<td></td>
<td></td>
<td>25</td>
<td></td>
<td>walking shoes
</td></tr>
<tr>
<td><a href="/wiki/Elven_boots" title="Elven boots">elven boots</a></td>
<td>8</td>
<td>15</td>
<td>1</td>
<td>15</td>
<td></td>
<td>leather</td>
<td>Stlth</td>
<td></td>
<td>12</td>
<td>Yes</td>
<td>*mud boots
</td></tr>
<tr>
<td><a href="/wiki/Kicking_boots" title="Kicking boots">kicking boots</a></td>
<td>8</td>
<td>50</td>
<td>1</td>
<td>50</td>
<td></td>
<td>iron</td>
<td>Kick</td>
<td></td>
<td>12</td>
<td>Yes</td>
<td>*buckled boots
</td></tr>
<tr>
<td><a href="/wiki/Fumble_boots" title="Fumble boots">fumble boots</a></td>
<td>30</td>
<td>20</td>
<td>1</td>
<td>20</td>
<td></td>
<td>leather</td>
<td>Fumble</td>
<td></td>
<td>12</td>
<td>Yes</td>
<td>*riding boots
</td></tr>
<tr>
<td><a href="/wiki/Levitation_boots" title="Levitation boots">levitation boots</a></td>
<td>30</td>
<td>15</td>
<td>1</td>
<td>15</td>
<td></td>
<td>leather</td>
<td>Lev</td>
<td></td>
<td>12</td>
<td>Yes</td>
<td>*snow boots
</td></tr>
<tr>
<td><a href="/wiki/Jumping_boots" title="Jumping boots">jumping boots</a></td>
<td>50</td>
<td>20</td>
<td>1</td>
<td>20</td>
<td></td>
<td>leather</td>
<td>Jump</td>
<td></td>
<td>12</td>
<td>Yes</td>
<td>*hiking boots
</td></tr>
<tr>
<td><a href="/wiki/Speed_boots" title="Speed boots">speed boots</a></td>
<td>50</td>
<td>20</td>
<td>1</td>
<td>20</td>
<td></td>
<td>leather</td>
<td>Speed</td>
<td></td>
<td>12</td>
<td>Yes</td>
<td>*combat boots
</td></tr>
<tr>
<td><a href="/wiki/Water_walking_boots" title="Water walking boots">water walking boots</a></td>
<td>50</td>
<td>20</td>
<td>1</td>
<td>20</td>
<td></td>
<td>leather</td>
<td>WWalk</td>
<td></td>
<td>12</td>
<td>Yes</td>
<td>*jungle boots
</td></tr>
<tr>
<td><a href="/wiki/High_boots" title="High boots">high boots</a></td>
<td>12</td>
<td>20</td>
<td>2</td>
<td>10</td>
<td></td>
<td>leather</td>
<td></td>
<td></td>
<td>15</td>
<td></td>
<td>jackboots
</td></tr>
<tr>
<td><a href="/wiki/Iron_shoes" title="Iron shoes">iron shoes</a></td>
<td>16</td>
<td>50</td>
<td>2</td>
<td>25</td>
<td></td>
<td>iron</td>
<td></td>
<td></td>
<td>7</td>
<td></td>
<td>hard shoes
</td></tr></tbody></table>
"""


"""
example file:

from .base_armour import Armour
from ... import material as M


class BaseShirt(Armour):
    pass


class HawaiianShirt(BaseShirt):

    def __init__(self):
        super().__init__('hawaiian shirt', weight=5, armour_class=0, material=M.Cloth)


class TShirt(BaseShirt):

    def __init__(self):
        super().__init__('t-shirt', weight=120, armour_class=0, material=M.Cloth)

"""


mydir = os.path.dirname(os.path.abspath(__file__))


def create_init(submodules):
    fname = os.path.join(mydir, '__init__.py')
    with open(fname, 'wt') as f:
        for m in submodules:
            f.write('from .{} import *\n'.format(m))


def get_class_name(name):
    name = name.replace('-', ' ')
    stripped = name
    capitalized = ''.join([word.capitalize() for word in stripped.split(' ')])
    return capitalized


def initialize(group):
    fname = os.path.join(mydir, '{}.py'.format(group.replace('-', '_').replace(' ', '_').lower()))
    class_name = 'Base' + get_class_name(group)
    if class_name[-1] == 's':  # make singular
        class_name = class_name[:-1]
    return [
        'from .base_armour import Armour',
        'from ... import material as M',
        '',
        '',
        'class {}(Armour):'.format(class_name),
        '    pass',
        '',
    ], class_name, fname


def get_class(base_class, attributes):
    class_name = get_class_name(attributes['name'])
    attributes['material'] = 'M.{}'.format(attributes['material'].capitalize())

    return [
        '',
        'class {}({}):'.format(class_name, base_class),
        '',
        '    def __init__(self):',
        '        super().__init__(\'{name}\', weight={weight}, armour_class={armour_class}, material={material})'.format(**attributes),
        '',
    ]


if __name__ == '__main__':
    doc = html.fromstring(TABLE)

    trs = doc.xpath('//tr')
    header = ['name', 'cost', 'weight', 'armour_class', 'weight_per_ac0', 'weight_per_ac5', 'material', 'effect', 'mc', 'prob', 'magical', 'appearance']

    current = {}

    print('processing {} rows'.format(len(trs)))

    all_files = []

    for row in trs[1:]:
        if len(row) == 1:
            # this is a group row
            group = row.text_content().strip()
            if current:
                content = '\n'.join(current['rows'])
                if current['base'] == 'BaseShirt':
                    with open(os.path.join(mydir, 'shirts.py')) as f:
                        expect = f.read()
                    import unittest
                    unittest.TestCase().assertEqual(expect, content)
                else:
                    with open(current['file'], 'wt') as f:
                        f.write(content)
            current['rows'], current['base'], current['file'] = initialize(group)
            all_files.append(current['file'])
        else:
            # this is an item row
            attributes = {k: v for k, v in zip(header, [t.text_content().strip() for t in row])}
            attributes['name'] = re.sub(r'(\(.+\))', '', attributes['name']).strip()
            if not attributes['magical']:
                # NOTE: we skip magical items because we don't implement the dynamics
                current['rows'] += get_class(current['base'], attributes)

    # write the last one
    content = '\n'.join(current['rows'])
    with open(current['file'], 'wt') as f:
        f.write(content)
    create_init(['base_armour'] + [os.path.basename(f).replace('.py', '') for f in all_files])
    print('done')

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python
"""
This file will generate the weapons from the Nethack wiki at
https://nethackwiki.com/wiki/Weapon
"""
import os
import re
from lxml import html


TABLE = """
<table class="prettytable striped">
<tbody><tr>
<th>Weapon Name</th>
<th>Cost</th>
<th>Weight</th>
<th>Prob&nbsp;(‰)</th>
<th colspan="2">Damage&nbsp;(S/L)</th>
<th>Material</th>
<th>Appearance</th>
<th>Tile</th>
<th>Glyph
</th></tr>
<tr>
<td colspan="10"><a href="/wiki/Daggers" class="mw-redirect" title="Daggers"><b>daggers</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Orcish_dagger" class="mw-redirect" title="Orcish dagger">orcish dagger</a></td>
<td>4 zm</td>
<td>10</td>
<td>12<sup>†</sup></td>
<td>d3</td>
<td>d3</td>
<td>iron</td>
<td>crude dagger</td>
<td><a href="/wiki/File:Orcish_dagger.png" class="image"><img alt="Orcish dagger.png" src="/mediawiki/images/8/8b/Orcish_dagger.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-darkgray">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Dagger" title="Dagger">dagger</a></td>
<td>4 zm</td>
<td>10</td>
<td>30</td>
<td>d4</td>
<td>d3</td>
<td>iron</td>
<td></td>
<td><a href="/wiki/File:Dagger.png" class="image"><img alt="Dagger.png" src="/mediawiki/images/4/45/Dagger.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Silver_dagger" class="mw-redirect" title="Silver dagger">silver dagger</a></td>
<td>40 zm</td>
<td>12</td>
<td>3</td>
<td>d4</td>
<td>d3</td>
<td>silver</td>
<td></td>
<td><a href="/wiki/File:Silver_dagger.png" class="image"><img alt="Silver dagger.png" src="/mediawiki/images/0/05/Silver_dagger.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-lightgray">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Athame" title="Athame">athame</a></td>
<td>4 zm</td>
<td>10</td>
<td>N/A</td>
<td>d4</td>
<td>d3</td>
<td>iron</td>
<td></td>
<td><a href="/wiki/File:Athame.png" class="image"><img alt="Athame.png" src="/mediawiki/images/0/02/Athame.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Elven_dagger" class="mw-redirect" title="Elven dagger">elven dagger</a></td>
<td>4 zm</td>
<td>10</td>
<td>10<sup>†</sup></td>
<td>d5</td>
<td>d3</td>
<td>wood</td>
<td>runed dagger</td>
<td><a href="/wiki/File:Elven_dagger.png" class="image"><img alt="Elven dagger.png" src="/mediawiki/images/3/3a/Elven_dagger.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-brown">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Knife" title="Knife"><b>knives</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Worm_tooth" title="Worm tooth">worm tooth</a></td>
<td>2 zm</td>
<td>20</td>
<td>N/A</td>
<td>d2</td>
<td>d2</td>
<td>undefined</td>
<td></td>
<td><a href="/wiki/File:Worm_tooth.png" class="image"><img alt="Worm tooth.png" src="/mediawiki/images/c/cb/Worm_tooth.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-white">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Knife" title="Knife">knife (shito)</a></td>
<td>4 zm</td>
<td>5</td>
<td>20</td>
<td>d3</td>
<td>d2</td>
<td>iron</td>
<td></td>
<td><a href="/wiki/File:Knife.png" class="image"><img alt="Knife.png" src="/mediawiki/images/5/58/Knife.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Stiletto" title="Stiletto">stiletto</a></td>
<td>4 zm</td>
<td>5</td>
<td>5</td>
<td>d3</td>
<td>d2</td>
<td>iron</td>
<td></td>
<td><a href="/wiki/File:Stiletto.png" class="image"><img alt="Stiletto.png" src="/mediawiki/images/f/f4/Stiletto.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Scalpel" title="Scalpel">scalpel</a></td>
<td>4 zm</td>
<td>5</td>
<td>N/A</td>
<td>d3</td>
<td>d3</td>
<td>metal</td>
<td></td>
<td><a href="/wiki/File:Scalpel.png" class="image"><img alt="Scalpel.png" src="/mediawiki/images/2/2c/Scalpel.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Crysknife" title="Crysknife">crysknife</a></td>
<td>100 zm</td>
<td>20</td>
<td>N/A</td>
<td>d10</td>
<td>d10</td>
<td>mineral</td>
<td></td>
<td><a href="/wiki/File:Crysknife.png" class="image"><img alt="Crysknife.png" src="/mediawiki/images/5/5a/Crysknife.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-white">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Axe" title="Axe"><b>axes</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Axe" title="Axe">axe</a></td>
<td>8 zm</td>
<td>60</td>
<td>40</td>
<td>d6</td>
<td>d4</td>
<td>iron</td>
<td></td>
<td><a href="/wiki/File:Axe.png" class="image"><img alt="Axe.png" src="/mediawiki/images/f/fb/Axe.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Battle-axe" title="Battle-axe">battle-axe</a></td>
<td>40 zm</td>
<td>120</td>
<td>10</td>
<td>d8+d4</td>
<td>d6+2d4</td>
<td>iron</td>
<td>double-headed axe</td>
<td><a href="/wiki/File:Battle-axe.png" class="image"><img alt="Battle-axe.png" src="/mediawiki/images/b/b2/Battle-axe.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Pick-axe" title="Pick-axe"><b>pick-axes</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Pick-axe" title="Pick-axe">pick-axe</a></td>
<td>50 zm</td>
<td>100</td>
<td>tool</td>
<td>d6</td>
<td>d3</td>
<td>iron</td>
<td></td>
<td><a href="/wiki/File:Pick-axe.png" class="image"><img alt="Pick-axe.png" src="/mediawiki/images/e/e6/Pick-axe.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">(</span>
</td></tr>
<tr>
<td><a href="/wiki/Dwarvish_mattock" title="Dwarvish mattock">dwarvish mattock</a></td>
<td>50 zm</td>
<td>120</td>
<td>13<sup>†</sup></td>
<td>d12</td>
<td>d8+2d6</td>
<td>iron</td>
<td>broad pick</td>
<td><a href="/wiki/File:Dwarvish_mattock.png" class="image"><img alt="Dwarvish mattock.png" src="/mediawiki/images/a/ae/Dwarvish_mattock.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Short_sword" title="Short sword"><b>short swords</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Orcish_short_sword" class="mw-redirect" title="Orcish short sword">orcish short sword</a></td>
<td>10 zm</td>
<td>30</td>
<td>3<sup>†</sup></td>
<td>d5</td>
<td>d8</td>
<td>iron</td>
<td>crude short sword</td>
<td><a href="/wiki/File:Orcish_short_sword.png" class="image"><img alt="Orcish short sword.png" src="/mediawiki/images/3/34/Orcish_short_sword.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-darkgray">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Short_sword" title="Short sword">short sword (wakizashi)</a></td>
<td>10 zm</td>
<td>30</td>
<td>8</td>
<td>d6</td>
<td>d8</td>
<td>iron</td>
<td></td>
<td><a href="/wiki/File:Short_sword.png" class="image"><img alt="Short sword.png" src="/mediawiki/images/3/35/Short_sword.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Dwarvish_short_sword" class="mw-redirect" title="Dwarvish short sword">dwarvish short sword</a></td>
<td>10 zm</td>
<td>30</td>
<td>2<sup>†</sup></td>
<td>d7</td>
<td>d8</td>
<td>iron</td>
<td>broad short sword</td>
<td><a href="/wiki/File:Dwarvish_short_sword.png" class="image"><img alt="Dwarvish short sword.png" src="/mediawiki/images/a/a4/Dwarvish_short_sword.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Elven_short_sword" class="mw-redirect" title="Elven short sword">elven short sword</a></td>
<td>10 zm</td>
<td>30</td>
<td>2<sup>†</sup></td>
<td>d8</td>
<td>d8</td>
<td>wood</td>
<td>runed short sword</td>
<td><a href="/wiki/File:Elven_short_sword.png" class="image"><img alt="Elven short sword.png" src="/mediawiki/images/5/57/Elven_short_sword.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-brown">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Broadsword" title="Broadsword"><b>broadswords</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Broadsword" title="Broadsword">broadsword (ninja-to)</a></td>
<td>10 zm</td>
<td>70</td>
<td>8</td>
<td>2d4</td>
<td>d6+1</td>
<td>iron</td>
<td></td>
<td><a href="/wiki/File:Broadsword.png" class="image"><img alt="Broadsword.png" src="/mediawiki/images/8/87/Broadsword.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Runesword" title="Runesword">runesword</a></td>
<td>300 zm</td>
<td>40</td>
<td>N/A</td>
<td>2d4</td>
<td>d6+1</td>
<td>iron</td>
<td>runed broadsword</td>
<td><a href="/wiki/File:Runesword.png" class="image"><img alt="Runesword.png" src="/mediawiki/images/b/b6/Runesword.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-darkgray">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Elven_broadsword" title="Elven broadsword">elven broadsword</a></td>
<td>10 zm</td>
<td>70</td>
<td>4<sup>†</sup></td>
<td>d6+d4</td>
<td>d6+1</td>
<td>wood</td>
<td>runed broadsword</td>
<td><a href="/wiki/File:Elven_broadsword.png" class="image"><img alt="Elven broadsword.png" src="/mediawiki/images/7/78/Elven_broadsword.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-brown">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Long_sword" title="Long sword"><b>long swords</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Long_sword" title="Long sword">long sword</a></td>
<td>15 zm</td>
<td>40</td>
<td>50<sup>†</sup></td>
<td>d8</td>
<td>d12</td>
<td>iron</td>
<td></td>
<td><a href="/wiki/File:Long_sword.png" class="image"><img alt="Long sword.png" src="/mediawiki/images/2/2d/Long_sword.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Katana" title="Katana">katana</a></td>
<td>80 zm</td>
<td>40</td>
<td>4</td>
<td>d10</td>
<td>d12</td>
<td>iron</td>
<td>samurai sword</td>
<td><a href="/wiki/File:Katana.png" class="image"><img alt="Katana.png" src="/mediawiki/images/e/e9/Katana.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Two-handed_sword" title="Two-handed sword"><b>two-handed swords</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Two-handed_sword" title="Two-handed sword">two-handed sword</a></td>
<td>50 zm</td>
<td>150</td>
<td>22</td>
<td>d12</td>
<td>3d6</td>
<td>iron</td>
<td></td>
<td><a href="/wiki/File:Two-handed_sword.png" class="image"><img alt="Two-handed sword.png" src="/mediawiki/images/d/d2/Two-handed_sword.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Tsurugi" title="Tsurugi">tsurugi</a></td>
<td>500 zm</td>
<td>60</td>
<td>N/A</td>
<td>d16</td>
<td>d8+2d6</td>
<td>metal</td>
<td>long samurai sword</td>
<td><a href="/wiki/File:Tsurugi.png" class="image"><img alt="Tsurugi.png" src="/mediawiki/images/c/c1/Tsurugi.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Scimitar" title="Scimitar"><b>scimitars</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Scimitar" title="Scimitar">scimitar</a></td>
<td>15 zm</td>
<td>40</td>
<td>15<sup>†</sup></td>
<td>d8</td>
<td>d8</td>
<td>iron</td>
<td>curved sword</td>
<td><a href="/wiki/File:Scimitar.png" class="image"><img alt="Scimitar.png" src="/mediawiki/images/e/e5/Scimitar.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Saber" class="mw-redirect" title="Saber"><b>sabers</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Silver_saber" title="Silver saber">silver saber</a></td>
<td>75 zm</td>
<td>40</td>
<td>6</td>
<td>d8</td>
<td>d8</td>
<td>silver</td>
<td></td>
<td><a href="/wiki/File:Silver_saber.png" class="image"><img alt="Silver saber.png" src="/mediawiki/images/7/74/Silver_saber.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-lightgray">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Club" title="Club"><b>clubs</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Club" title="Club">club</a></td>
<td>3 zm</td>
<td>30</td>
<td>12</td>
<td>d6</td>
<td>d3</td>
<td>wood</td>
<td></td>
<td><a href="/wiki/File:Club.png" class="image"><img alt="Club.png" src="/mediawiki/images/5/50/Club.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-brown">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Aklys" title="Aklys">aklys</a></td>
<td>4 zm</td>
<td>15</td>
<td>8</td>
<td>d6</td>
<td>d3</td>
<td>iron</td>
<td>thonged club</td>
<td><a href="/wiki/File:Aklys.png" class="image"><img alt="Aklys.png" src="/mediawiki/images/2/2c/Aklys.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Mace" title="Mace"><b>maces</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Mace" title="Mace">mace</a></td>
<td>5 zm</td>
<td>30</td>
<td>40</td>
<td>d6+1</td>
<td>d6</td>
<td>iron</td>
<td></td>
<td><a href="/wiki/File:Mace.png" class="image"><img alt="Mace.png" src="/mediawiki/images/6/63/Mace.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Morning_star" title="Morning star"><b>morning stars</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Morning_star" title="Morning star">morning star</a></td>
<td>10 zm</td>
<td>120</td>
<td>12</td>
<td>2d4</td>
<td>d6+1</td>
<td>iron</td>
<td></td>
<td><a href="/wiki/File:Morning_star.png" class="image"><img alt="Morning star.png" src="/mediawiki/images/f/fd/Morning_star.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Flail" title="Flail"><b>flails</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Flail" title="Flail">flail (nunchaku)</a></td>
<td>4 zm</td>
<td>15</td>
<td>40</td>
<td>d6+1</td>
<td>2d4</td>
<td>iron</td>
<td></td>
<td><a href="/wiki/File:Flail.png" class="image"><img alt="Flail.png" src="/mediawiki/images/1/1d/Flail.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Grappling_hook" title="Grappling hook">grappling hook</a></td>
<td>50 zm</td>
<td>30</td>
<td>tool</td>
<td>d2</td>
<td>d6</td>
<td>iron</td>
<td>iron hook</td>
<td><a href="/wiki/File:Grappling_hook.png" class="image"><img alt="Grappling hook.png" src="/mediawiki/images/4/43/Grappling_hook.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">(</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Hammer" class="mw-redirect" title="Hammer"><b>hammers</b></a>
</td></tr>
<tr>
<td><a href="/wiki/War_hammer" title="War hammer">war hammer</a></td>
<td>5 zm</td>
<td>50</td>
<td>15</td>
<td>d4+1</td>
<td>d4</td>
<td>iron</td>
<td></td>
<td><a href="/wiki/File:War_hammer.png" class="image"><img alt="War hammer.png" src="/mediawiki/images/a/ab/War_hammer.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Quarterstaff" title="Quarterstaff"><b>quarterstaves</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Quarterstaff" title="Quarterstaff">quarterstaff</a></td>
<td>5 zm</td>
<td>40</td>
<td>11<sup>†</sup></td>
<td>d6</td>
<td>d6</td>
<td>wood</td>
<td>staff</td>
<td><a href="/wiki/File:Quarterstaff.png" class="image"><img alt="Quarterstaff.png" src="/mediawiki/images/8/85/Quarterstaff.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-brown">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Polearms" class="mw-redirect" title="Polearms"><b>polearms</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Partisan" class="mw-redirect" title="Partisan">partisan</a></td>
<td>10 zm</td>
<td>80</td>
<td>5</td>
<td>d6</td>
<td>d6+1</td>
<td>iron</td>
<td>vulgar polearm</td>
<td><a href="/wiki/File:Partisan.png" class="image"><img alt="Partisan.png" src="/mediawiki/images/c/c1/Partisan.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Fauchard" class="mw-redirect" title="Fauchard">fauchard</a></td>
<td>5 zm</td>
<td>60</td>
<td>6</td>
<td>d6</td>
<td>d8</td>
<td>iron</td>
<td>pole sickle</td>
<td><a href="/wiki/File:Fauchard.png" class="image"><img alt="Fauchard.png" src="/mediawiki/images/5/5a/Fauchard.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Glaive" class="mw-redirect" title="Glaive">glaive (naginata)</a></td>
<td>6 zm</td>
<td>75</td>
<td>8</td>
<td>d6</td>
<td>d10</td>
<td>iron</td>
<td>single-edged polearm</td>
<td><a href="/wiki/File:Glaive.png" class="image"><img alt="Glaive.png" src="/mediawiki/images/e/e9/Glaive.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Bec-de-corbin" class="mw-redirect" title="Bec-de-corbin">bec-de-corbin</a></td>
<td>8 zm</td>
<td>100</td>
<td>4</td>
<td>d8</td>
<td>d6</td>
<td>iron</td>
<td>beaked polearm</td>
<td><a href="/wiki/File:Bec_de_corbin.png" class="image"><img alt="Bec de corbin.png" src="/mediawiki/images/c/cb/Bec_de_corbin.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Spetum" class="mw-redirect" title="Spetum">spetum</a></td>
<td>5 zm</td>
<td>50</td>
<td>5</td>
<td>d6+1</td>
<td>2d6</td>
<td>iron</td>
<td>forked polearm</td>
<td><a href="/wiki/File:Spetum.png" class="image"><img alt="Spetum.png" src="/mediawiki/images/4/4f/Spetum.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Lucern_hammer" class="mw-redirect" title="Lucern hammer">lucern hammer</a></td>
<td>7 zm</td>
<td>150</td>
<td>5</td>
<td>2d4</td>
<td>d6</td>
<td>iron</td>
<td>pronged polearm</td>
<td><a href="/wiki/File:Lucern_hammer.png" class="image"><img alt="Lucern hammer.png" src="/mediawiki/images/9/95/Lucern_hammer.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Guisarme" class="mw-redirect" title="Guisarme">guisarme</a></td>
<td>5 zm</td>
<td>80</td>
<td>6</td>
<td>2d4</td>
<td>d8</td>
<td>iron</td>
<td>pruning hook</td>
<td><a href="/wiki/File:Guisarme.png" class="image"><img alt="Guisarme.png" src="/mediawiki/images/0/07/Guisarme.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Ranseur" class="mw-redirect" title="Ranseur">ranseur</a></td>
<td>6 zm</td>
<td>50</td>
<td>5</td>
<td>2d4</td>
<td>2d4</td>
<td>iron</td>
<td>hilted polearm</td>
<td><a href="/wiki/File:Ranseur.png" class="image"><img alt="Ranseur.png" src="/mediawiki/images/b/b9/Ranseur.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Voulge" class="mw-redirect" title="Voulge">voulge</a></td>
<td>5 zm</td>
<td>125</td>
<td>4</td>
<td>2d4</td>
<td>2d4</td>
<td>iron</td>
<td>pole cleaver</td>
<td><a href="/wiki/File:Voulge.png" class="image"><img alt="Voulge.png" src="/mediawiki/images/7/7e/Voulge.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Bill-guisarme" class="mw-redirect" title="Bill-guisarme">bill-guisarme</a></td>
<td>7 zm</td>
<td>120</td>
<td>4</td>
<td>2d4</td>
<td>d10</td>
<td>iron</td>
<td>hooked polearm</td>
<td><a href="/wiki/File:Bill-guisarme.png" class="image"><img alt="Bill-guisarme.png" src="/mediawiki/images/5/5d/Bill-guisarme.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Bardiche" class="mw-redirect" title="Bardiche">bardiche</a></td>
<td>7 zm</td>
<td>120</td>
<td>4</td>
<td>2d4</td>
<td>3d4</td>
<td>iron</td>
<td>long poleaxe</td>
<td><a href="/wiki/File:Bardiche.png" class="image"><img alt="Bardiche.png" src="/mediawiki/images/4/4d/Bardiche.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Halberd" class="mw-redirect" title="Halberd">halberd</a></td>
<td>10 zm</td>
<td>150</td>
<td>8</td>
<td>d10</td>
<td>2d6</td>
<td>iron</td>
<td>angled poleaxe</td>
<td><a href="/wiki/File:Halberd.png" class="image"><img alt="Halberd.png" src="/mediawiki/images/1/12/Halberd.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Spear" title="Spear"><b>spears</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Orcish_spear" class="mw-redirect" title="Orcish spear">orcish spear</a></td>
<td>3 zm</td>
<td>30</td>
<td>13<sup>†</sup></td>
<td>d5</td>
<td>d8</td>
<td>iron</td>
<td>crude spear</td>
<td><a href="/wiki/File:Orcish_spear.png" class="image"><img alt="Orcish spear.png" src="/mediawiki/images/2/28/Orcish_spear.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-darkgray">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Spear" title="Spear">spear</a></td>
<td>3 zm</td>
<td>30</td>
<td>50<sup>†</sup></td>
<td>d6</td>
<td>d8</td>
<td>iron</td>
<td></td>
<td><a href="/wiki/File:Spear.png" class="image"><img alt="Spear.png" src="/mediawiki/images/4/4f/Spear.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Silver_spear" class="mw-redirect" title="Silver spear">silver spear</a></td>
<td>40 zm</td>
<td>36</td>
<td>2</td>
<td>d6</td>
<td>d8</td>
<td>silver</td>
<td></td>
<td><a href="/wiki/File:Silver_spear.png" class="image"><img alt="Silver spear.png" src="/mediawiki/images/c/ca/Silver_spear.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-lightgray">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Elven_spear" class="mw-redirect" title="Elven spear">elven spear</a></td>
<td>3 zm</td>
<td>30</td>
<td>10<sup>†</sup></td>
<td>d7</td>
<td>d8</td>
<td>wood</td>
<td>runed spear</td>
<td><a href="/wiki/File:Elven_spear.png" class="image"><img alt="Elven spear.png" src="/mediawiki/images/6/66/Elven_spear.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-brown">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Dwarvish_spear" class="mw-redirect" title="Dwarvish spear">dwarvish spear</a></td>
<td>3 zm</td>
<td>35</td>
<td>12<sup>†</sup></td>
<td>d8</td>
<td>d8</td>
<td>iron</td>
<td>stout spear</td>
<td><a href="/wiki/File:Dwarvish_spear.png" class="image"><img alt="Dwarvish spear.png" src="/mediawiki/images/d/da/Dwarvish_spear.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Javelin" title="Javelin">javelin</a></td>
<td>3 zm</td>
<td>20</td>
<td>10</td>
<td>d6</td>
<td>d6</td>
<td>iron</td>
<td>throwing spear</td>
<td><a href="/wiki/File:Javelin.png" class="image"><img alt="Javelin.png" src="/mediawiki/images/1/1d/Javelin.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Trident" title="Trident"><b>tridents</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Trident" title="Trident">trident</a></td>
<td>5 zm</td>
<td>25</td>
<td>8</td>
<td>d6+1</td>
<td>3d4</td>
<td>iron</td>
<td></td>
<td><a href="/wiki/File:Trident.png" class="image"><img alt="Trident.png" src="/mediawiki/images/9/9a/Trident.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Lance" title="Lance"><b>lances</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Lance" title="Lance">lance</a> (1)</td>
<td>10 zm</td>
<td>180</td>
<td>4</td>
<td>d6</td>
<td>d8</td>
<td>iron</td>
<td></td>
<td><a href="/wiki/File:Lance.png" class="image"><img alt="Lance.png" src="/mediawiki/images/f/f8/Lance.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Bow" title="Bow"><b>bows</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Orcish_bow" title="Orcish bow">orcish bow</a></td>
<td>60 zm</td>
<td>30</td>
<td>12<sup>†</sup></td>
<td>d2</td>
<td>d2</td>
<td>wood</td>
<td>crude bow</td>
<td><a href="/wiki/File:Orcish_bow.png" class="image"><img alt="Orcish bow.png" src="/mediawiki/images/c/cb/Orcish_bow.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-darkgray">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Orcish_arrow" title="Orcish arrow">orcish arrow</a></td>
<td>2 zm</td>
<td>1</td>
<td>20<sup>†</sup></td>
<td>d5</td>
<td>d6</td>
<td>iron</td>
<td>crude arrow</td>
<td><a href="/wiki/File:Orcish_arrow.png" class="image"><img alt="Orcish arrow.png" src="/mediawiki/images/a/a4/Orcish_arrow.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-darkgray">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Bow" title="Bow">bow</a></td>
<td>60 zm</td>
<td>30</td>
<td>24</td>
<td>d2</td>
<td>d2</td>
<td>wood</td>
<td></td>
<td><a href="/wiki/File:Bow.png" class="image"><img alt="Bow.png" src="/mediawiki/images/6/65/Bow.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-brown">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Arrow" title="Arrow">arrow</a></td>
<td>2 zm</td>
<td>1</td>
<td>55<sup>†</sup></td>
<td>d6</td>
<td>d6</td>
<td>iron</td>
<td></td>
<td><a href="/wiki/File:Arrow.png" class="image"><img alt="Arrow.png" src="/mediawiki/images/4/41/Arrow.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Elven_bow" title="Elven bow">elven bow</a></td>
<td>60 zm</td>
<td>30</td>
<td>12<sup>†</sup></td>
<td>d2</td>
<td>d2</td>
<td>wood</td>
<td>runed bow</td>
<td><a href="/wiki/File:Elven_bow.png" class="image"><img alt="Elven bow.png" src="/mediawiki/images/f/fd/Elven_bow.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-brown">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Elven_arrow" title="Elven arrow">elven arrow</a></td>
<td>2 zm</td>
<td>1</td>
<td>20<sup>†</sup></td>
<td>d7</td>
<td>d6</td>
<td>wood</td>
<td>runed arrow</td>
<td><a href="/wiki/File:Elven_arrow.png" class="image"><img alt="Elven arrow.png" src="/mediawiki/images/4/4a/Elven_arrow.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-brown">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Yumi" title="Yumi">yumi</a></td>
<td>60 zm</td>
<td>30</td>
<td>0</td>
<td>d2</td>
<td>d2</td>
<td>wood</td>
<td>long bow</td>
<td><a href="/wiki/File:Yumi.png" class="image"><img alt="Yumi.png" src="/mediawiki/images/f/fb/Yumi.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-brown">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Ya" title="Ya">ya</a></td>
<td>4 zm</td>
<td>1</td>
<td>15</td>
<td>d7</td>
<td>d7</td>
<td>metal</td>
<td>bamboo arrow</td>
<td><a href="/wiki/File:Bamboo_arrow.png" class="image"><img alt="Bamboo arrow.png" src="/mediawiki/images/d/de/Bamboo_arrow.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Silver_arrow" title="Silver arrow">silver arrow</a></td>
<td>5 zm</td>
<td>1</td>
<td>12</td>
<td>d6</td>
<td>d6</td>
<td>silver</td>
<td></td>
<td><a href="/wiki/File:Silver_arrow.png" class="image"><img alt="Silver arrow.png" src="/mediawiki/images/c/cd/Silver_arrow.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-lightgray">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Sling" title="Sling"><b>slings</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Sling" title="Sling">sling</a></td>
<td>20 zm</td>
<td>3</td>
<td>40</td>
<td>d2</td>
<td>d2</td>
<td>leather</td>
<td></td>
<td><a href="/wiki/File:Sling.png" class="image"><img alt="Sling.png" src="/mediawiki/images/9/93/Sling.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-brown">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Flintstone" class="mw-redirect" title="Flintstone">flintstone</a></td>
<td>1 zm</td>
<td>10</td>
<td>gem</td>
<td>d6</td>
<td>d6</td>
<td>mineral</td>
<td>gray stone</td>
<td><a href="/wiki/File:Gray_stone.png" class="image"><img alt="Gray stone.png" src="/mediawiki/images/a/a0/Gray_stone.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-lightgray">*</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Crossbow" title="Crossbow"><b>crossbows</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Crossbow" title="Crossbow">crossbow</a></td>
<td>40 zm</td>
<td>50</td>
<td>45<sup>†</sup></td>
<td>d2</td>
<td>d2</td>
<td>wood</td>
<td></td>
<td><a href="/wiki/File:Crossbow.png" class="image"><img alt="Crossbow.png" src="/mediawiki/images/7/79/Crossbow.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-brown">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Crossbow_bolt" title="Crossbow bolt">crossbow bolt</a></td>
<td>2 zm</td>
<td>1</td>
<td>55<sup>†</sup></td>
<td>d4+1</td>
<td>d6+1</td>
<td>iron</td>
<td></td>
<td><a href="/wiki/File:Crossbow_bolt.png" class="image"><img alt="Crossbow bolt.png" src="/mediawiki/images/e/e5/Crossbow_bolt.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Dart" title="Dart"><b>darts</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Dart" title="Dart">dart</a></td>
<td>2 zm</td>
<td>1</td>
<td>60<sup>†</sup></td>
<td>d3</td>
<td>d2</td>
<td>iron</td>
<td></td>
<td><a href="/wiki/File:Dart.png" class="image"><img alt="Dart.png" src="/mediawiki/images/2/29/Dart.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Shuriken" title="Shuriken"><b>shurikens</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Shuriken" title="Shuriken">shuriken</a></td>
<td>5 zm</td>
<td>1</td>
<td>35</td>
<td>d8</td>
<td>d6</td>
<td>iron</td>
<td>throwing star</td>
<td><a href="/wiki/File:Shuriken.png" class="image"><img alt="Shuriken.png" src="/mediawiki/images/a/ac/Shuriken.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-cyan">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Boomerang" title="Boomerang"><b>boomerangs</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Boomerang" title="Boomerang">boomerang</a></td>
<td>20 zm</td>
<td>5</td>
<td>15</td>
<td>d9</td>
<td>d9</td>
<td>wood</td>
<td></td>
<td><a href="/wiki/File:Boomerang.png" class="image"><img alt="Boomerang.png" src="/mediawiki/images/7/7c/Boomerang.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-brown">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Whip" class="mw-redirect" title="Whip"><b>whips</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Bullwhip" title="Bullwhip">bullwhip</a></td>
<td>4 zm</td>
<td>20</td>
<td>2</td>
<td>d2</td>
<td>1</td>
<td>leather</td>
<td></td>
<td><a href="/wiki/File:Bullwhip.png" class="image"><img alt="Bullwhip.png" src="/mediawiki/images/4/44/Bullwhip.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-brown">)</span>
</td></tr>
<tr>
<td><a href="/wiki/Rubber_hose" title="Rubber hose">rubber hose</a></td>
<td>3 zm</td>
<td>20</td>
<td>N/A</td>
<td>d4</td>
<td>d3</td>
<td>plastic</td>
<td></td>
<td><a href="/wiki/File:Rubber_hose.png" class="image"><img alt="Rubber hose.png" src="/mediawiki/images/2/26/Rubber_hose.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-brown">)</span>
</td></tr>
<tr>
<td colspan="10"><a href="/wiki/Unicorn_horn" title="Unicorn horn"><b>unicorn horns</b></a>
</td></tr>
<tr>
<td><a href="/wiki/Unicorn_horn" title="Unicorn horn">unicorn horn</a></td>
<td>100 zm</td>
<td>20</td>
<td>tool</td>
<td>d12</td>
<td>d12</td>
<td>bone</td>
<td></td>
<td><a href="/wiki/File:Unicorn_horn.png" class="image"><img alt="Unicorn horn.png" src="/mediawiki/images/6/65/Unicorn_horn.png" width="16" height="16"></a></td>
<td><span class="nhsym clr-white">(</span>
</td></tr></tbody></table>
"""


"""
example file:

from .base_weapon import Weapon
from ... import dice as D, material as M


class BaseDagger(Weapon):
    pass


class OrcishDagger(BaseDagger):

    def __init__(self):
        super().__init__('orcish dagger', weight=10, damage=D.SingleDice(3), material=M.Iron)
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
    cleaned = re.sub(r'(\(.+\))', '', capitalized)
    return cleaned


def initialize(group):
    fname = os.path.join(mydir, '{}.py'.format(group.replace('-', '_').replace(' ', '_').lower()))
    class_name = 'Base' + get_class_name(group)
    if class_name[-1] == 's':  # make singular
        class_name = class_name[:-1]
    return [
        'from .base_weapon import Weapon',
        'from ... import dice as D, material as M',
        '',
        '',
        'class {}(Weapon):'.format(class_name),
        '    pass',
        '',
    ], class_name, fname


def get_class(base_class, attributes):
    class_name = get_class_name(attributes['name'])
    attributes['material'] = 'M.{}'.format(attributes['material'].capitalize())
    attributes['damage'] = 'D.Dice.from_str(\'{}\')'.format(attributes['damage'])

    return [
        '',
        'class {}({}):'.format(class_name, base_class),
        '',
        '    def __init__(self):',
        '        super().__init__(\'{name}\', weight={weight}, damage={damage}, material={material}, hit={hit})'.format(**attributes),
        '',
    ]


if __name__ == '__main__':
    doc = html.fromstring(TABLE)

    trs = doc.xpath('//tr')
    header = ['name', 'cost', 'weight', 'prob', 'damage_s', 'damage', 'material', 'appearance']

    current = {}

    print('processing {} rows'.format(len(trs)))

    all_files = []

    for row in trs[1:]:
        if len(row) == 1:
            # this is a group row
            group = row.text_content().strip()
            if current:
                content = '\n'.join(current['rows'])
                if current['base'] == 'BaseDagger':
                    with open(os.path.join(mydir, 'daggers.py')) as f:
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
            if current['base'] == 'BaseDagger':
                attributes['hit'] = 2
            elif current['base'] == 'BaseKnive':
                attributes['hit'] = 1
            else:
                attributes['hit'] = 0
            current['rows'] += get_class(current['base'], attributes)

    # write the last one
    content = '\n'.join(current['rows'])
    with open(current['file'], 'wt') as f:
        f.write(content)
    create_init(['base_weapon', 'unarmed'] + [os.path.basename(f).replace('.py', '') for f in all_files])
    print('done')

from sgrf.algorithms.adithya_rajesh.adithya_rajesh import AdithyaRajesh
from sgrf.algorithms.chang_chen.chang_chen import ChangChen
from sgrf.algorithms.eid_schwenker.eid_schwenker import EidSchwenker
from sgrf.algorithms.gupta_jaafar.gupta_jaafar import GuptaJaafar
from sgrf.algorithms.islam_hossain_andersson.islam_hossain_andersson import IslamHossainAndersson
from sgrf.algorithms.joshi_kumar.joshi_kumar import JoshiKumar
from sgrf.algorithms.maung.maung import Maung
from sgrf.algorithms.mohanty_rambhatla.mohanty_rambhatla import MohantyRambhatla
from sgrf.algorithms.mohmmad_dadi.mohmmad_dadi import MohmmadDadi
from sgrf.algorithms.murthy_jadon.murthy_jadon import MurthyJadon
from sgrf.algorithms.naidoo_omlin.naidoo_omlin import NaidooOmlin
from sgrf.algorithms.nguyen_huynh.nguyen_huynh import NguyenHuynh
from sgrf.algorithms.oyedotun_khashman.oyedotun_khashman import OyedotunKhashman
from sgrf.algorithms.pinto_borges.pinto_borges import PintoBorges
from sgrf.algorithms.zhuang_yang.zhuang_yang import ZhuangYang
from sgrf.data.algorithm import ALGORITHM

ALGORITHM_FUNCTIONS = {
    ALGORITHM.MURTHY_JADON: MurthyJadon(),
    ALGORITHM.MAUNG: Maung(),
    ALGORITHM.ADITHYA_RAJESH: AdithyaRajesh(),
    ALGORITHM.EID_SCHWENKER: EidSchwenker(),
    ALGORITHM.ISLAM_HOSSAIN_ANDERSSON: IslamHossainAndersson(),
    ALGORITHM.PINTO_BORGES: PintoBorges(),
    ALGORITHM.MOHMMAD_DADI: MohmmadDadi(),
    ALGORITHM.GUPTA_JAAFAR: GuptaJaafar(),
    ALGORITHM.MOHANTY_RAMBHATLA: MohantyRambhatla(),
    ALGORITHM.ZHUANG_YANG: ZhuangYang(),
    ALGORITHM.CHANG_CHEN: ChangChen(),
    ALGORITHM.NAIDOO_OMLIN: NaidooOmlin(),
    ALGORITHM.JOSHI_KUMAR: JoshiKumar(),
    ALGORITHM.NGUYEN_HUYNH: NguyenHuynh(),
    ALGORITHM.OYEDOTUN_KHASHMAN: OyedotunKhashman(),
}

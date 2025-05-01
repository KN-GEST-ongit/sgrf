from bdgs.algorithms.adithya_rajesh.adithya_rajesh import AdithyaRajesh
from bdgs.algorithms.eid_schwenker.eid_schwenker import EidSchwenker
from bdgs.algorithms.islam_hossain_andersson.islam_hossain_andersson import IslamHossainAndersson
from bdgs.algorithms.maung.maung import Maung
from bdgs.algorithms.mohmmad_dadi.mohmmad_dadi import MohmmadDadi
from bdgs.algorithms.murthy_jadon.murthy_jadon import MurthyJadon
from bdgs.algorithms.pinto_borges.pinto_borges import PintoBorges
from bdgs.algorithms.gupta_jaafar.gupta_jaafar import GuptaJaafar


from bdgs.data.algorithm import ALGORITHM

ALGORITHM_FUNCTIONS = {
    ALGORITHM.MURTHY_JADON: MurthyJadon(),
    ALGORITHM.MAUNG: Maung(),
    ALGORITHM.ADITHYA_RAJESH: AdithyaRajesh(),
    ALGORITHM.EID_SCHWENKER: EidSchwenker(),
    ALGORITHM.ISLAM_HOSSAIN_ANDERSSON: IslamHossainAndersson(),
    ALGORITHM.PINTO_BORGES: PintoBorges(),
    ALGORITHM.MOHMMAD_DADI: MohmmadDadi(),
    ALGORITHM.GUPTA_JAAFAR: GuptaJaafar(),
}

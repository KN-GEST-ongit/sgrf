from bdgs.algorithms.adithya_rajesh.adithya_rajesh_learning_data import AdithyaRajeshLearningData
from bdgs.algorithms.gupta_jaafar.gupta_jaafar_learning_data import GuptaJaafarLearningData
from bdgs.algorithms.islam_hossain_andersson.islam_hossain_andersson_learning_data import \
    IslamHossainAnderssonLearningData
from bdgs.algorithms.maung.maung_learning_data import MaungLearningData
from bdgs.algorithms.murthy_jadon.murthy_jadon_learning_data import MurthyJadonLearningData
from bdgs.algorithms.pinto_borges.pinto_borges_learning_data import PintoBorgesLearningData
from bdgs.algorithms.mohanty_rambhatla.mohanty_rambhatla_learning_data import MohantyRambhatlaLearningData
from bdgs.algorithms.zhuang_yang.zhuang_yang_learning_data import ZhuangYangLearningData
from bdgs.algorithms.chang_chen.chang_chen_learning_data import ChangChenLearningData
from bdgs.algorithms.joshi_kumar.joshi_kumar_learning_data import JoshiKumarLearningData
from bdgs.data.algorithm import ALGORITHM
from bdgs.data.gesture import GESTURE
from bdgs.models.learning_data import LearningData
from scripts.file_coords_parser import parse_etiquette, parse_file_coords


def choose_learning_data(algorithm: ALGORITHM, image_path: str, bg_image_path: str, etiquette: str):
    label = GESTURE(parse_etiquette(etiquette))
    if algorithm == ALGORITHM.MURTHY_JADON:
        return MurthyJadonLearningData(image_path=image_path, bg_image_path=bg_image_path, label=label)
    elif algorithm == ALGORITHM.PINTO_BORGES:
        return PintoBorgesLearningData(image_path=image_path, coords=parse_file_coords(etiquette), label=label)
    elif algorithm == ALGORITHM.ADITHYA_RAJESH:
        return AdithyaRajeshLearningData(image_path=image_path, coords=parse_file_coords(etiquette), label=label)
    elif algorithm == ALGORITHM.ISLAM_HOSSAIN_ANDERSSON:
        return IslamHossainAnderssonLearningData(image_path=image_path, bg_image_path=bg_image_path,
                                                 coords=parse_file_coords(etiquette), label=label)
    elif algorithm == ALGORITHM.GUPTA_JAAFAR:
        return GuptaJaafarLearningData(image_path=image_path, coords=parse_file_coords(etiquette), label=label)
    elif algorithm == ALGORITHM.MAUNG:
        return MaungLearningData(image_path=image_path, coords=parse_file_coords(etiquette), label=label)
    elif algorithm == ALGORITHM.MOHANTY_RAMBHATLA:
            return MohantyRambhatlaLearningData(image_path=image_path, coords=parse_file_coords(etiquette), label=label)
    elif algorithm == ALGORITHM.ZHUANG_YANG:
            return ZhuangYangLearningData(image_path=image_path, coords=parse_file_coords(etiquette), label=label)
    elif algorithm == ALGORITHM.CHANG_CHEN:
            return ChangChenLearningData(image_path=image_path, coords=parse_file_coords(etiquette), label=label)
    elif algorithm == ALGORITHM.JOSHI_KUMAR:
            return JoshiKumarLearningData(image_path=image_path, coords=parse_file_coords(etiquette), label=label)
    else:
        return LearningData(image_path=image_path, label=label)

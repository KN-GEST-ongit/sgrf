from bdgs.algorithms.adithya_rajesh.adithya_rajesh_payload import AdithyaRajeshPayload
from bdgs.algorithms.gupta_jaafar.gupta_jaafar_payload import GuptaJaafarPayload
from bdgs.algorithms.islam_hossain_andersson.islam_hossain_andersson_payload import IslamHossainAnderssonPayload
from bdgs.algorithms.maung.maung_payload import MaungPayload
from bdgs.algorithms.murthy_jadon.murthy_jadon_payload import MurthyJadonPayload
from bdgs.algorithms.pinto_borges.pinto_borges_payload import PintoBorgesPayload
from bdgs.algorithms.zhuang_yang.zhuang_yang_payload import ZhuangYangPayload
from bdgs.data.algorithm import ALGORITHM
from bdgs.models.image_payload import ImagePayload


def choose_payload(algorithm, background, coords, image):
    if algorithm == ALGORITHM.MURTHY_JADON:
        payload = MurthyJadonPayload(image=image, bg_image=background)
    elif algorithm == ALGORITHM.ISLAM_HOSSAIN_ANDERSSON:
        payload = IslamHossainAnderssonPayload(image=image, bg_image=background, coords=coords)
    elif algorithm == ALGORITHM.PINTO_BORGES:
        payload = PintoBorgesPayload(image=image, coords=coords)
    elif algorithm == ALGORITHM.ADITHYA_RAJESH:
        payload = AdithyaRajeshPayload(image=image, coords=coords)
    elif algorithm == ALGORITHM.GUPTA_JAAFAR:
        payload = GuptaJaafarPayload(image=image, coords=coords)
    elif algorithm == ALGORITHM.MAUNG:
        payload = MaungPayload(image=image, coords=coords)
    elif algorithm == ALGORITHM.ZHUANG_YANG:
        payload = ZhuangYangPayload(image=image, coords=coords)
    else:
        payload = ImagePayload(image=image)
    return payload

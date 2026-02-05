from sgrf.algorithms.adithya_rajesh.adithya_rajesh_payload import AdithyaRajeshPayload
from sgrf.algorithms.chang_chen.chang_chen_payload import ChangChenPayload
from sgrf.algorithms.gupta_jaafar.gupta_jaafar_payload import GuptaJaafarPayload
from sgrf.algorithms.islam_hossain_andersson.islam_hossain_andersson_payload import IslamHossainAnderssonPayload
from sgrf.algorithms.joshi_kumar.joshi_kumar_payload import JoshiKumarPayload
from sgrf.algorithms.maung.maung_payload import MaungPayload
from sgrf.algorithms.mohanty_rambhatla.mohanty_rambhatla_payload import MohantyRambhatlaPayload
from sgrf.algorithms.murthy_jadon.murthy_jadon_payload import MurthyJadonPayload
from sgrf.algorithms.nguyen_huynh.nguyen_huynh_payload import NguyenHuynhPayload
from sgrf.algorithms.oyedotun_khashman.oyedotun_khashman_payload import OyedotunKhashmanPayload
from sgrf.algorithms.pinto_borges.pinto_borges_payload import PintoBorgesPayload
from sgrf.algorithms.zhuang_yang.zhuang_yang_payload import ZhuangYangPayload
from sgrf.data.algorithm import ALGORITHM
from sgrf.models.image_payload import ImagePayload


def choose_payload(algorithm, background, coords, image):
    if algorithm == ALGORITHM.MURTHY_JADON:
        payload = MurthyJadonPayload(image=image, bg_image=background)
        if background is None:
            raise ValueError(f"Algorithm {algorithm} requires background image")
    elif algorithm == ALGORITHM.ISLAM_HOSSAIN_ANDERSSON:
        if background is None:
            raise ValueError(f"Algorithm {algorithm} requires background image")
        payload = IslamHossainAnderssonPayload(image=image, bg_image=background, coords=coords)
    elif algorithm == ALGORITHM.PINTO_BORGES:
        payload = PintoBorgesPayload(image=image, coords=coords)
    elif algorithm == ALGORITHM.ADITHYA_RAJESH:
        payload = AdithyaRajeshPayload(image=image, coords=coords)
    elif algorithm == ALGORITHM.GUPTA_JAAFAR:
        payload = GuptaJaafarPayload(image=image, coords=coords)
    elif algorithm == ALGORITHM.MAUNG:
        payload = MaungPayload(image=image, coords=coords)
    elif algorithm == ALGORITHM.MOHANTY_RAMBHATLA:
        payload = MohantyRambhatlaPayload(image=image, coords=coords)
    elif algorithm == ALGORITHM.ZHUANG_YANG:
        payload = ZhuangYangPayload(image=image, coords=coords)
    elif algorithm == ALGORITHM.CHANG_CHEN:
        payload = ChangChenPayload(image=image, coords=coords)
    elif algorithm == ALGORITHM.JOSHI_KUMAR:
        payload = JoshiKumarPayload(image=image, coords=coords)
    elif algorithm == ALGORITHM.NGUYEN_HUYNH:
        payload = NguyenHuynhPayload(image=image, coords=coords)
    elif algorithm == ALGORITHM.OYEDOTUN_KHASHMAN:
        payload = OyedotunKhashmanPayload(image=image, coords=coords)
    else:
        payload = ImagePayload(image=image)
    return payload

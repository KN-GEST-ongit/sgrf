import os
import posixpath
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from scripts.loaders.base_loader import BaseDatasetLoader
from scripts.nextcloud_cache import get_nextcloud_cache
from scripts.nextcloud_client import get_concurrency, get_nextcloud_client
from scripts.vars import NEXTCLOUD_TRAINING_IMAGES_PATH, TRAINING_IMAGES_PATH


class SGRFDatasetLoader(BaseDatasetLoader):
    def get_learning_files(skip_empty=True, shuffle=True, limit=None, offset=0,
                           limit_recordings_of_single_person_single_gesture=None,
                           limit_images_in_single_person_single_recording=None,
                           limit_people=None, base_path=TRAINING_IMAGES_PATH, source="local"):
        if source == "nextcloud":
            if base_path == TRAINING_IMAGES_PATH:
                base_path = NEXTCLOUD_TRAINING_IMAGES_PATH
            return SGRFDatasetLoader.get_learning_files_nextcloud(
                skip_empty=skip_empty, shuffle=shuffle, limit=limit, offset=offset,
                limit_recordings_of_single_person_single_gesture=limit_recordings_of_single_person_single_gesture,
                limit_images_in_single_person_single_recording=limit_images_in_single_person_single_recording,
                limit_people=limit_people, base_path=base_path,
            )
        if source != "local":
            raise ValueError(f"Unknown source {source!r}, expected 'local' or 'nextcloud'")

        image_files = []
        classify_file = None

        visited_paths = {}
        visited_people = {}

        with tqdm(desc="Local: scanning dataset") as bar:
            for root, _, files in os.walk(base_path):
                bar.update(1)
                if ".git" in root:
                    continue
                for file in files:
                    if file.lower().endswith(".txt"):
                        classify_file = os.path.join(root, file)
                        break
                if classify_file is None or len(files) == 0: continue

                root = Path(root).resolve()
                parent_path = root.parent
                people_path = parent_path.parents[1].name
                visited_paths[parent_path] = visited_paths.get(parent_path, 0) + 1
                visited_people[people_path] = visited_people.get(people_path, 0) + 1

                if limit_recordings_of_single_person_single_gesture is not None and visited_paths.get(parent_path,
                                                                                                      0) > limit_recordings_of_single_person_single_gesture:
                    continue
                if limit_people is not None and len(visited_people) > limit_people:
                    break

                with open(classify_file, "r") as f:
                    classify_row = [line.split("\n")[0] for line in f]
                files = sorted(files)
                files.pop(0)
                bg_image = files[0]

                added = 0
                for index in range(len(files) - 1):
                    if files[index].lower().endswith(('.png', '.jpg', '.jpeg')):
                        if limit_images_in_single_person_single_recording is not None and added >= limit_images_in_single_person_single_recording:
                            break
                        if skip_empty:
                            if classify_row[index].split(" ")[0] != "0":
                                image_files.append(
                                    (os.path.join(root, files[index]), classify_row[index], (os.path.join(root, bg_image))))
                                added += 1
                        else:
                            image_files.append(
                                (os.path.join(root, files[index]), classify_row[index], (os.path.join(root, bg_image))))
                            added += 1

        if shuffle: random.shuffle(image_files)
        return image_files[offset:(limit + offset if limit is not None else None)]

    def get_learning_files_nextcloud(skip_empty=True, shuffle=True, limit=None, offset=0,
                                     limit_recordings_of_single_person_single_gesture=None,
                                     limit_images_in_single_person_single_recording=None,
                                     limit_people=None, base_path=NEXTCLOUD_TRAINING_IMAGES_PATH):
        client = get_nextcloud_client()
        cache = get_nextcloud_cache()

        with tqdm(desc="Nextcloud: walking dataset tree") as bar:
            walk_results = client.walk_parallel(base_path, on_progress=bar.update)

        classify_file = None
        visited_paths = {}
        visited_people = {}
        pending = []

        for root, dir_entries, file_entries in walk_results:
            if ".git" in root:
                continue
            entry_by_name = {e.name: e for e in file_entries}
            for name, entry in entry_by_name.items():
                if name.lower().endswith(".txt"):
                    classify_file = entry.path
                    break
            if classify_file is None or len(file_entries) == 0: continue

            parent_path = posixpath.dirname(root)
            grandparent_path = posixpath.dirname(posixpath.dirname(parent_path))
            people_path = posixpath.basename(grandparent_path)
            visited_paths[parent_path] = visited_paths.get(parent_path, 0) + 1
            visited_people[people_path] = visited_people.get(people_path, 0) + 1

            if limit_recordings_of_single_person_single_gesture is not None and visited_paths.get(parent_path,
                                                                                                  0) > limit_recordings_of_single_person_single_gesture:
                continue
            if limit_people is not None and len(visited_people) > limit_people:
                break

            pending.append((root, entry_by_name, classify_file))

        unique_classify_files = {p[2] for p in pending}
        with tqdm(desc="Nextcloud: reading classification files", total=len(unique_classify_files)) as bar:
            classify_contents = client.read_text_many(unique_classify_files, on_progress=bar.update)

        image_files = []
        known_sizes = {}

        for root, entry_by_name, classify_file in pending:
            classify_row = classify_contents[classify_file].splitlines()
            files = sorted(entry_by_name.keys())
            files.pop(0)
            bg_image = files[0]
            bg_path = posixpath.join(root, bg_image)
            known_sizes[bg_path] = entry_by_name[bg_image].size

            added = 0
            for index in range(len(files) - 1):
                if files[index].lower().endswith(('.png', '.jpg', '.jpeg')):
                    if limit_images_in_single_person_single_recording is not None and added >= limit_images_in_single_person_single_recording:
                        break
                    image_path = posixpath.join(root, files[index])
                    if skip_empty:
                        if classify_row[index].split(" ")[0] != "0":
                            image_files.append((image_path, classify_row[index], bg_path))
                            known_sizes[image_path] = entry_by_name[files[index]].size
                            added += 1
                    else:
                        image_files.append((image_path, classify_row[index], bg_path))
                        known_sizes[image_path] = entry_by_name[files[index]].size
                        added += 1

        if shuffle: random.shuffle(image_files)
        selected = image_files[offset:(limit + offset if limit is not None else None)]

        remote_paths = {path for entry in selected for path in (entry[0], entry[2])}

        total_needed_bytes = sum(known_sizes.get(path, 0) for path in remote_paths)
        if total_needed_bytes > cache.max_size_bytes:
            print(
                f"[nextcloud] warning: this call needs ~{total_needed_bytes / (1024 * 1024):.0f}MB of images, "
                f"but the cache is capped at {cache.max_size_bytes / (1024 * 1024):.0f}MB "
                "(NEXTCLOUD_CACHE_MAX_SIZE_MB). Files already read by an earlier part of this same batch may get "
                "evicted before a later part reads them, which will make cv2.imread(...) return None and crash. "
                "Raise NEXTCLOUD_CACHE_MAX_SIZE_MB in .env to at least the size printed above, or reduce `limit`."
            )

        def _resolve(remote_path):
            return remote_path, cache.get(client, remote_path, size_hint=known_sizes.get(remote_path))

        local_path_by_remote = {}
        with tqdm(desc="Nextcloud: downloading images", total=len(remote_paths)) as bar:
            with ThreadPoolExecutor(max_workers=min(get_concurrency(), len(remote_paths) or 1)) as pool:
                futures = [pool.submit(_resolve, remote_path) for remote_path in remote_paths]
                for future in as_completed(futures):
                    remote_path, local_path = future.result()
                    local_path_by_remote[remote_path] = local_path
                    bar.update(1)

        return [
            (local_path_by_remote[image_path], classify_row, local_path_by_remote[bg_path])
            for image_path, classify_row, bg_path in selected
        ]

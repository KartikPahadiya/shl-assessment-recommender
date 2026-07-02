def parse_duration(duration_str):
    if not duration_str:
        return None

    try:
        num = int(duration_str.split()[0])
        return num
    except:
        return None


def metadata_filter(catalog, constraints):
    filtered = []

    for item in catalog:
        keep = True

        # ---------- Language ----------
        language = constraints.get("language")
        if language:
            langs = [x.lower() for x in item.get("languages", [])]

            if language.lower() not in " ".join(langs):
                keep = False

        # ---------- Duration ----------
        max_duration = constraints.get("duration")
        if max_duration:
            duration = parse_duration(item.get("duration", ""))

            if duration is not None and duration > max_duration:
                keep = False

        # ---------- Remote ----------
        remote = constraints.get("remote")
        if remote is True:
            if item.get("remote", "").lower() != "yes":
                keep = False

        # ---------- Adaptive ----------
        adaptive = constraints.get("adaptive")
        if adaptive is True:
            if item.get("adaptive", "").lower() != "yes":
                keep = False

        if keep:
            filtered.append(item)

    return filtered
import datetime as dt
from typing import Any, Optional, Sequence, Union

_strptime_formats = [
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d",
]


def dt_utc(*args: Any) -> dt.datetime:
    return dt.datetime(*args).replace(tzinfo=dt.timezone.utc)


def dt_utc_from_str(string: str, string_format: Optional[str] = None) -> dt.datetime:
    dtp = None

    if string_format:
        try:
            dtp = dt.datetime.strptime(string, string_format)
        except ValueError:
            for fmt in _strptime_formats:
                try:
                    dtp = dt.datetime.strptime(string, fmt)
                except ValueError:
                    pass
    else:
        for fmt in _strptime_formats:
            try:
                dtp = dt.datetime.strptime(string, fmt)
            except ValueError:
                pass

    if dtp:
        if not dtp.tzinfo:
            dtp = dtp.replace(tzinfo=dt.timezone.utc)
        return dtp

    raise ValueError('could not convert "{0!s}", unkown format'.format(string))


def sanitize_dt(
    dtp: Union[str, dt.datetime, Sequence[str], Sequence[dt.datetime]]
) -> Union[dt.datetime, Sequence[dt.datetime]]:
    
    if isinstance(dtp, dt.datetime) and dtp.tzinfo is None:
        return dtp.replace(tzinfo=dt.timezone.utc)
    elif isinstance(dtp, dt.datetime) and dtp.tzinfo != dt.timezone.utc:
        return dtp.astimezone(dt.timezone.utc)
    elif isinstance(dtp, str):
        return dt_utc_from_str(dtp)
    elif hasattr(dtp, "__iter__"):
        _dtp = list(dtp)

        if isinstance(_dtp[0], dt.datetime):
            for i in range(len(_dtp)):
                if _dtp[i].tzinfo is None:
                    _dtp[i] = _dtp[i].replace(tzinfo=dt.timezone.utc)
                else:
                    _dtp[i] = _dtp[i].astimezone(dt.timezone.utc)
        elif isinstance(_dtp[0], str):
            _dtp = [dt_utc_from_str(_) for _ in _dtp]

        return _dtp
    else:
        return dtp
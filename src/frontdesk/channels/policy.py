"""A channel's policy narrows the business pack (ADR-0010, M4 design D5): the channel's turns run
on a copy of the pack with the features the channel switches off turned off."""

from __future__ import annotations

from frontdesk.tenancy.pack import BusinessPack, ChannelPolicy


def narrow(pack: BusinessPack, policy: ChannelPolicy) -> BusinessPack:
    features = pack.features.model_copy(
        update={
            "orientation": pack.features.orientation and policy.orientation,
            "order_status": pack.features.order_status and policy.order_status,
        }
    )
    integrations = pack.integrations
    calendar = integrations.calendar
    if calendar is not None:
        booking = calendar.chat_booking and policy.chat_booking
        integrations = integrations.model_copy(
            update={"calendar": calendar.model_copy(update={"chat_booking": booking})}
        )
    return pack.model_copy(update={"features": features, "integrations": integrations})

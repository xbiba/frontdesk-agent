# Runbook — receptionist agent

- **Document answers fall back to "no information" too often:** read the turn trace (`check_reasons`). `quote_not_in_source` and `unsupported_claim` mean a sentence was not backed by its passage; `number_not_in_quote` and `price_in_answer` mean a number or price did not come from the passage; `link_not_in_quote` means a sentence carried a web address, email address or markdown link that its quote does not contain; `clarify_not_offered` means the model asked for something other than a service or a doctor, the only questions its prompt offers; an empty source list means retrieval found nothing — check the active knowledge version and the retrieval evaluation.
- **Wrong or outdated facts:** prices, hours and contacts are written from records. Roll back knowledge with `frontdesk knowledge rollback --env-file deploy/<business>.env`, fix the business pack, then draft, review, approve and activate a new version.
- **Classifier rejections spike:** read the `reason` field in the logs (`unknown_service_id`, `text_has_clinical_terms`, …); a new service needs a knowledge version, not a prompt change.
- **The deployment refuses to start after an embedder change:** run `frontdesk knowledge reembed --env-file deploy/<business>.env`.
- **Stop the bot:** until the M3 kill switch exists, stop the process; conversations stay visible to staff on Instagram.

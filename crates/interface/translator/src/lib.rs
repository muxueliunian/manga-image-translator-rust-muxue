use std::borrow::Cow;

pub use aio_translator::*;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct LanguageWrapper(pub Language);

impl JsonSchema for LanguageWrapper {
    fn schema_name() -> Cow<'static, str> {
        "LanguageWrapper".into()
    }

    fn schema_id() -> Cow<'static, str> {
        concat!(module_path!(), "::", "LanguageWrapper").into()
    }
    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        {
            let mut map = serde_json::Map::new();
            map.insert(
                "oneOf".into(),
                serde_json::Value::Array({
                    let mut enum_values = Vec::new();
                    enum_values.push(to_enum_schema("cht", "Chinese Traditional"));
                    enum_values.push(to_enum_schema("chs", "Chinese Simplified"));
                    for lang in Language::all() {
                        let name = lang.to_name().unwrap();
                        if let Some(code) = lang.to_639_1() {
                            enum_values.push(to_enum_schema(code, name));
                        }
                        if let Some(code) = lang.to_639_3() {
                            enum_values.push(to_enum_schema(code, name));
                        }
                    }
                    enum_values
                }),
            );
            schemars::Schema::from(map)
        }
    }
}

fn to_enum_schema(name: &str, desc: &str) -> Value {
    use schemars::_private::{
        get_title_and_description, insert_metadata_property_if_nonempty, new_unit_enum_variant,
    };
    let mut schema = new_unit_enum_variant(name);
    let (title, desc): (&str, &str) = get_title_and_description(desc);

    insert_metadata_property_if_nonempty(&mut schema, "title", title);
    insert_metadata_property_if_nonempty(&mut schema, "description", desc);
    schema.to_value()
}

impl<'de> Deserialize<'de> for LanguageWrapper {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct EnumVisitor;

        impl<'de> serde::de::Visitor<'de> for EnumVisitor {
            type Value = LanguageWrapper;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "a string matching one of the enum variants")
            }

            fn visit_str<E>(self, value: &str) -> Result<LanguageWrapper, E>
            where
                E: serde::de::Error,
            {
                let lang = value.trim().to_lowercase();
                if lang == "cht" {
                    return Ok(LanguageWrapper(Language::ChineseTraditional));
                } else if lang == "chs" {
                    return Ok(LanguageWrapper(Language::Chinese));
                }
                (if lang.len() == 2 {
                    Language::from_639_1(&lang)
                } else {
                    Language::from_639_3(&lang)
                })
                .map(LanguageWrapper)
                .ok_or_else(|| E::custom(format!("invalid lang code: \"{}\"", value)))
            }
        }

        deserializer.deserialize_str(EnumVisitor)
    }
}

impl Serialize for LanguageWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let str = self
            .0
            .to_639_1()
            .or(self.0.to_639_3())
            .unwrap_or_else(|| self.0.to_name().unwrap());
        serializer.serialize_str(str)
    }
}
